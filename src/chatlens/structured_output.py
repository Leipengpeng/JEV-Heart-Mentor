"""Offline JSON schema filtering for MLX sampling; no text or logits logged."""

import json
import os
import time

_TOKEN_DATA = {}


def _record_private_failure(generated, exception):
    """Optional on-device reproduction data; never decode or log this prefix."""
    destination = os.environ.get("CHATLENS_JSON_DIAGNOSTICS")
    if not destination or not os.path.isabs(destination):
        return
    try:
        record = {"generated_token_ids": generated, "error_type": type(exception).__name__}
        if isinstance(exception, RuntimeError) and str(exception).startswith("JSON "):
            record["error_reason"] = str(exception)
        fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            os.fchmod(fd, 0o600)
            os.write(fd, (json.dumps(record) + "\n").encode("utf-8"))
        finally:
            os.close(fd)
    except OSError:
        pass


def make_json_processor(tokenizer, schema):
    import mlx.core as mx
    from lmformatenforcer import JsonSchemaParser, TokenEnforcer, TokenEnforcerTokenizerData
    from lmformatenforcer.characterlevelparser import CharacterLevelParserConfig
    from lmformatenforcer.tokenlist import TokenList

    started = time.perf_counter()
    key = id(tokenizer)
    if key not in _TOKEN_DATA:
        # Use the generic API: the package's Transformers adapter imports
        # a tokenizer class removed by this environment's Transformers 5.
        vocabulary = tokenizer.get_vocab()
        # This checkpoint's chat end token is an EOS in MLX's wrapper but is
        # absent from Transformers' all_special_ids. It must never be treated
        # as ordinary free text, or decoding would stop inside a JSON string.
        special = set(tokenizer.all_special_ids) | set(tokenizer.eos_token_ids)
        anchor = tokenizer.encode("0", add_special_tokens=False)[-1]
        regular = []
        for token in sorted(vocabulary.values()):
            if token in special:
                continue
            text = tokenizer.decode([anchor, token])[1:]
            standalone = tokenizer.decode([token])
            regular.append((token, text, len(text) > len(standalone)))
        data = TokenEnforcerTokenizerData(
            regular,
            lambda ids: tokenizer.decode(ids).rstrip("�"),
            list(tokenizer.eos_token_ids),
            False,
            max(vocabulary.values()) + 1,
        )
        # The package uses only the trie root (token-initial characters) as its
        # alphabet. Include characters elsewhere in decoded vocabulary tokens.
        # Byte-level tokens can still combine into additional Unicode characters;
        # those are admitted during strict character advancement below.
        data.tokenizer_alphabet = "".join(
            sorted({character for _, text, _ in regular for character in text if ord(character) >= 32})
        )
        _TOKEN_DATA[key] = data
    data = _TOKEN_DATA[key]
    # Set the alphabet before constructing the JSON parser: its free-text
    # context is initialized once and must include this Chinese vocabulary.
    parser = JsonSchemaParser(schema, config=CharacterLevelParserConfig(alphabet=data.tokenizer_alphabet))
    prompt_length = None
    eos = set(tokenizer.eos_token_ids)
    stats = {
        "setup_seconds": 0.0,
        "filter_seconds": 0.0,
        "steps": 0,
        "unicode_characters_added": 0,
        "parser_failures": 0,
    }

    class StrictTokenEnforcer(TokenEnforcer):
        def _compute_allowed_tokens(self, state_tokens, state):
            # Upstream catches unexpected errors, logs the decoded private
            # prefix, and permits EOS. Fail without logging instead: an internal
            # failure must never masquerade as a completed JSON result.
            try:
                cache_key = state.parser.cache_key()
                if cache_key is not None and cache_key in self.allowed_token_cache:
                    state.allowed_tokens = self.allowed_token_cache[cache_key]
                    return
                allowed = TokenList(self.use_bitmask, self.vocab_size)
                self._collect_allowed_tokens(
                    state.parser, self.tokenizer_tree.root, allowed, state.parser.shortcut_key()
                )
                if state.parser.can_end():
                    allowed.extend(list(eos))
                if not allowed:
                    raise ValueError("No allowed JSON continuation")
                state.allowed_tokens = allowed
                if cache_key is not None:
                    self.allowed_token_cache[cache_key] = allowed
            except Exception as exc:
                raise RuntimeError(f"JSON constraint computation failed ({type(exc).__name__})") from None

        def _admit_unicode(self, current_parser, character):
            # UTF-8 byte fragments decode to a full character only after several
            # tokens. A free-text shortcut may permit those fragments even when
            # that character has never appeared in any individually decoded token.
            # JsonSchemaParser silently empties its stack for unknown characters;
            # expand the free-text alphabet before validating/consuming them.
            if ord(character) < 32 or 0xD800 <= ord(character) <= 0xDFFF:
                return
            parsers = [self.root_parser, current_parser]
            seen = set()
            added = False
            while parsers:
                item = parsers.pop()
                if id(item) in seen:
                    continue
                seen.add(id(item))
                parsers.extend(getattr(item, "parsers", ()))
                if not isinstance(item, JsonSchemaParser):
                    continue
                if character not in item.config.alphabet:
                    item.config.alphabet += character
                    added = True
                if character != '"' and character not in item.context.alphabet_without_quotes:
                    item.context.alphabet_without_quotes += character
                # Primitive stack entries can refer to an earlier parser clone.
                parsers.extend(entry.root for entry in item.object_stack if hasattr(entry, "root"))
            if added:
                stats["unicode_characters_added"] += 1

        def _apply_new_characters(self, state, token_sequence):
            token = token_sequence[-1]
            if token not in state.allowed_tokens.allowed_tokens:
                raise RuntimeError("JSON generation received a disallowed token")
            next_state = self.OutputTensorState(parser=state.parser)
            if token in eos:
                # MLX may prefetch another logits step after sampling EOS. The
                # termination marker is not JSON text; retain the finished state
                # rather than feeding its decoded special-token spelling to it.
                if not state.parser.can_end():
                    raise RuntimeError("JSON generation received EOS before completion")
                next_state.current_word_tokens = state.current_word_tokens
                return next_state
            if token in self.tokenizer_tree.new_word_tokens:
                next_state.current_word_tokens = [token]
                characters = self.tokenizer_tree.tokens_to_strs[token]
            else:
                next_state.current_word_tokens = state.current_word_tokens + [token]
                before = self.decoder(state.current_word_tokens)
                after = self.decoder(next_state.current_word_tokens)
                if not after.startswith(before):
                    raise RuntimeError("JSON token decoding changed an accepted prefix")
                characters = after[len(before) :]
            for character in characters:
                self._admit_unicode(next_state.parser, character)
                if character not in next_state.parser.get_allowed_characters():
                    raise RuntimeError("JSON generation received a disallowed character")
                next_state.parser = next_state.parser.add_character(character)
            return next_state

    enforcer = StrictTokenEnforcer(data, parser)
    stats["setup_seconds"] = time.perf_counter() - started

    def processor(tokens, logits):
        nonlocal prompt_length
        tick = time.perf_counter()
        if prompt_length is None:
            prompt_length = tokens.size
        generated = tokens[prompt_length:].tolist()
        stats["steps"] += 1
        try:
            allowed = enforcer.get_allowed_tokens(generated).allowed_tokens
            # Confirm real decoded JSON completeness independently of the parser
            # before preferring EOS. This also guards against library regressions.
            stops = eos.intersection(allowed)
            if stops:
                try:
                    completed = generated
                    while completed and completed[-1] in eos:
                        completed = completed[:-1]
                    json.loads(tokenizer.decode(completed))
                except (ValueError, TypeError):
                    raise RuntimeError("JSON constraint requested EOS before a complete value") from None
                allowed = sorted(stops)
            if not allowed or any(token < 0 or token >= logits.shape[-1] for token in allowed):
                raise RuntimeError("JSON format constraint produced no valid token")
            mask = mx.full((logits.shape[-1],), -float("inf"), dtype=logits.dtype)
            mask[mx.array(allowed, dtype=mx.int32)] = 0
            return logits + mask
        except Exception as exc:
            stats["parser_failures"] += 1
            _record_private_failure(generated, exc)
            raise
        finally:
            stats["filter_seconds"] += time.perf_counter() - tick

    return processor, stats
