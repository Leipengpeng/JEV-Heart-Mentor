"""Public, authored synthetic preview. No models, account or private state."""

import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .config import PACKAGE_DIR


def demo_data():
    scenes = [
        (
            "小晴",
            [
                ("self", "周六我们去看展吧，我来订票。"),
                ("other", "好呀，我想看那个插画展。"),
                ("self", "下午两点在展馆门口见？"),
                ("other", "两点有点赶，三点怎么样？"),
                ("self", "三点没问题。"),
                ("other", "那就交给你啦，别又把自己落在家里。"),
            ],
            [
                (
                    "对方已经兴冲冲加入计划，还把想看的展览圈出来了。",
                    ["应允", "期待", "追问"],
                    ["开心", "期待", "平静"],
                    "应下插画展，顺手把出发时间定下来。",
                ),
                (
                    "对方在认真调整时间，三点是这轮对话的新约定。",
                    ["应允", "商量", "支招"],
                    ["平静", "关心", "期待"],
                    "确认三点见，把订票和见面地点一起安排好。",
                ),
                (
                    "对方放心把安排交给你，末尾还轻轻逗了你一下。",
                    ["调侃", "包揽", "期待"],
                    ["开心", "期待", "关心"],
                    "回一句轻松玩笑，再确认三点在展馆门口见。",
                ),
            ],
        ),
        (
            "阿澄",
            [
                ("other", "明天要面试了，感觉准备得还是不够。"),
                ("self", "你已经练了一周，项目经历也顺过好多遍。"),
                ("other", "我最怕一紧张，把自己做过的项目全忘了。"),
                ("self", "要不要我当一次面试官，陪你过一遍？"),
                ("other", "好，那你问轻一点，我现在有点怂。"),
            ],
            [
                (
                    "对方临近面试有点慌，这会儿很想有人帮忙稳稳心神。",
                    ["安慰", "打气", "陪伴"],
                    ["焦虑", "期待", "疑惑"],
                    "先肯定这一周的准备，再问最担心哪一部分。",
                ),
                (
                    "对方把紧张点说具体了，卡住的是现场讲项目的信心。",
                    ["陪伴", "支招", "打气"],
                    ["焦虑", "疑惑", "期待"],
                    "陪对方用三句话复述一个最熟悉的项目。",
                ),
                (
                    "对方答应练习又撒了点娇，希望模拟面试温柔开场。",
                    ["陪伴", "玩笑", "应允"],
                    ["期待", "焦虑", "开心"],
                    "从最熟悉的项目问起，先让对方顺利讲完一次。",
                ),
            ],
        ),
    ]
    contacts = {}
    for name, rows, analyses in scenes:
        messages, results, turn = [], {}, 0
        for index, (role, text) in enumerate(rows):
            messages.append(
                {"id": f"synthetic-{name}-{index}", "timestamp": 1735689600 + index * 60, "role": role, "text": text}
            )
            if role == "other":
                analysis, actions, emotions, detail = analyses[turn]
                ranked = [
                    {
                        "id": f"demo-action-{j}",
                        "action": title,
                        "score": score,
                        "score10": score * 10,
                        "rank": j + 1,
                        "kind": "reply",
                        "detail": detail if j == 0 else "结合当前话题，选择这种互动方式。",
                        "reply": "",
                    }
                    for j, (title, score) in enumerate(zip(actions, (0.86, 0.57, 0.24)))
                ]
                results[index] = {
                    "demo": True,
                    "proposal": {"analysis": analysis, "options": ranked},
                    "decision": {"ranked": ranked, "model": "synthetic preset"},
                    "emotion": {
                        "available": True,
                        "ranked": [
                            {"label": label, "score": score} for label, score in zip(emotions, (0.78, 0.52, 0.21))
                        ],
                    },
                    "card": {"next_step": detail},
                }
                turn += 1
        contacts[name] = {"messages": messages, "results": results}
    return {"contacts": contacts}


class DemoHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.headers.get("Host", "").split(":")[0] not in ("127.0.0.1", "localhost"):
            self.send_error(403)
            return
        if self.path in ("/", "/plugin"):
            body = (PACKAGE_DIR / "web/cards.html").read_text()
            data = json.dumps(demo_data(), ensure_ascii=False).replace("<", "\\u003c")
            body = body.replace(
                "<header>", '<aside class="demo-banner">合成演示 · 文案和评分均为预置示例</aside><header>'
            )
            body = body.replace(
                "<script>window.CARD_TOKEN=", "<script>window.CARD_PREVIEW=" + data + ";window.CARD_TOKEN="
            )
            mime = "text/html; charset=utf-8"
        elif self.path in ("/cards.css", "/cards.js"):
            body = (PACKAGE_DIR / "web" / self.path[1:]).read_text()
            mime = "text/css; charset=utf-8" if self.path.endswith(".css") else "text/javascript; charset=utf-8"
        else:
            self.send_error(404)
            return
        payload = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(payload)


def main(port=0, open_browser=False):
    with ThreadingHTTPServer(("127.0.0.1", port), DemoHandler) as server:
        url = f"http://127.0.0.1:{server.server_port}"
        print(f"ChatLens 合成演示：{url}", flush=True)
        if open_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
