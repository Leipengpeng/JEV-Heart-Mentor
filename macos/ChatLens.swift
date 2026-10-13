import Cocoa
import WebKit
import ScreenCaptureKit
import Vision

extension CGRect { var area: CGFloat { isNull ? 0 : width * height } }

func dockPlacement(wechat: NSRect, available: NSRect) -> (frame: NSRect, side: String) {
    let width: CGFloat = 350, height = min(620, available.height)
    let right = wechat.maxX + 12, left = wechat.minX - 12 - width
    let side = right + width <= available.maxX ? "right" : (left >= available.minX ? "left" : "none")
    let x = side == "right" ? right : (side == "left" ? left : max(available.minX,min(right,available.maxX-width)))
    let y = max(available.minY,min(wechat.maxY-height,available.maxY-height))
    return (NSRect(x:x,y:y,width:width,height:height),side)
}

// A valid selection opens the panel. Focus changes, failed captures and an
// temporarily empty window list must never undo that explicit user action.
struct DecisionPanelVisibility {
    var enabled = true
    var hasSelectedMessage = false
    var revision = ""
    var dismissedAt: TimeInterval?
    var contactRevision = ""
    var visible: Bool { enabled && hasSelectedMessage }

    mutating func receive(_ selection: [String: Any]?) {
        guard let selected = selection, let status = selected["status"] as? String,
              ["matched","self_message","unknown_role"].contains(status),
              let message = selected["message"] as? [String: Any],
              let id = message["id"] as? String, !id.isEmpty,
              let nextRevision = selected["revision"] as? String, !nextRevision.isEmpty else { return }
        hasSelectedMessage = true
        guard nextRevision != revision else { return }
        revision = nextRevision
        // A status request already in flight when the user closes the panel
        // cannot reopen it. A genuinely later selection can.
        let selectedAt = selected["selected_at"] as? TimeInterval ?? 0
        if dismissedAt.map({ selectedAt > $0 }) ?? true { enabled = true }
    }

    // An observed chat opens an empty consent card before any message has
    // been imported or analysed. Repeated status polls cannot undo a close.
    mutating func receiveContact(_ value: [String: Any]) {
        guard let nextRevision = value["contact_revision"] as? String, !nextRevision.isEmpty,
              let observedAt = value["contact_observed_at"] as? TimeInterval, observedAt > 0,
              let displayName = value["display_name"] as? String, !displayName.isEmpty else { return }
        hasSelectedMessage = true
        guard nextRevision != contactRevision else { return }
        contactRevision = nextRevision
        if dismissedAt.map({ observedAt > $0 }) ?? true { enabled = true }
    }

    mutating func dismiss(at time: TimeInterval = Date().timeIntervalSince1970) {
        enabled = false
        dismissedAt = time
    }

    mutating func toggle() {
        if enabled { dismiss() } else { enabled = true }
    }
}

func retainWechatWindow(observed: (pid_t, CGRect)?, previous: (pid_t, CGRect)?) -> (pid_t, CGRect)? {
    observed ?? previous
}

let processEnv = ProcessInfo.processInfo.environment
let projectRoot = URL(fileURLWithPath: CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : FileManager.default.currentDirectoryPath)
let supportRoot = URL(fileURLWithPath: CommandLine.arguments.count > 2 ? CommandLine.arguments[2] : (processEnv["CHATLENS_DATA_DIR"] ?? FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Application Support/ChatLensOSS").path))
let pythonExecutable = URL(fileURLWithPath: CommandLine.arguments.count > 3 ? CommandLine.arguments[3] : (processEnv["CHATLENS_PYTHON"] ?? projectRoot.appendingPathComponent(".venv/bin/python").path))
let buildRoot = URL(fileURLWithPath: CommandLine.arguments.count > 4 ? CommandLine.arguments[4] : (processEnv["CHATLENS_BUILD_DIR"] ?? projectRoot.appendingPathComponent("build").path))
let diagnosticsRoot = ProcessInfo.processInfo.environment["CHATLENS_DIAGNOSTICS"].map { URL(fileURLWithPath: $0) } ?? supportRoot.appendingPathComponent("diagnostics")
let screenshotRoot = ProcessInfo.processInfo.environment["CHATLENS_SCREENSHOTS"].map { URL(fileURLWithPath: $0) } ?? supportRoot.appendingPathComponent("screenshots")

func jsonString(_ object: Any) -> String {
    guard let data = try? JSONSerialization.data(withJSONObject: object, options: [.fragmentsAllowed]),
          let string = String(data: data, encoding: .utf8) else { return "null" }
    return string
}

func normalizedName(_ text: String) -> String {
    let trimmed = text.replacingOccurrences(of: "\\s+", with: "", options: .regularExpression)
    return trimmed.replacingOccurrences(of: "[（(][0-9]+[）)]$", with: "", options: .regularExpression)
}

func recognize(_ image: CGImage) throws -> [[String: Any]] {
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.recognitionLanguages = ["zh-Hans", "zh-Hant", "en-US"]
    request.usesLanguageCorrection = true
    try VNImageRequestHandler(cgImage: image, options: [:]).perform([request])
    return (request.results ?? []).compactMap { observation in
        guard let text = observation.topCandidates(1).first else { return nil }
        let box = observation.boundingBox
        return ["text": text.string, "confidence": text.confidence,
                "x": Double(box.minX), "y": Double(1 - box.maxY),
                "w": Double(box.width), "h": Double(box.height)]
    }.sorted { ($0["y"] as! Double) < ($1["y"] as! Double) }
}

func captureWechat(target: String) async throws -> [String: Any] {
    if !CGPreflightScreenCaptureAccess() {
        return ["ok": false, "code": "permission", "error": "需要为「微信对话助手」开启屏幕录制权限。点击权限按钮后，在系统设置中开启，再重新打开本应用。"]
    }
    let content = try await SCShareableContent.excludingDesktopWindows(true, onScreenWindowsOnly: true)
    let windows = content.windows.filter {
        $0.owningApplication?.bundleIdentifier == "com.tencent.xinWeChat" &&
        $0.frame.width > 600 && $0.frame.height > 400 && $0.windowLayer == 0
    }
    guard let window = windows.max(by: { $0.frame.width * $0.frame.height < $1.frame.width * $1.frame.height }) else {
        return ["ok": false, "code": "no_window", "error": "没有找到可见的微信主窗口。请打开微信和指定联系人的对话。"]
    }
    let filter = SCContentFilter(desktopIndependentWindow: window)
    let config = SCStreamConfiguration()
    config.width = Int(window.frame.width * 2)
    config.height = Int(window.frame.height * 2)
    config.showsCursor = false
    config.ignoreShadowsSingleWindow = true
    let image = try await SCScreenshotManager.captureImage(contentFilter: filter, configuration: config)
    return try analyzeWechatImage(image, target: target)
}

func analyzeWechatImage(_ image: CGImage, target: String) throws -> [String: Any] {
    let width = CGFloat(image.width), height = CGFloat(image.height)
    let headerHeight = min(height * 0.11, 140)
    guard let headerImage = image.cropping(to: CGRect(x: 0, y: 0, width: width, height: headerHeight)) else {
        throw NSError(domain: "ChatLens", code: 1, userInfo: [NSLocalizedDescriptionKey: "无法读取微信窗口标题"])
    }
    let headerLines = try recognize(headerImage)
    // Match the conversation heading, never a name from the sidebar message list.
    let heading = headerLines.first { line in
        let x = line["x"] as! Double
        let y = line["y"] as! Double
        return x > 0.2 && y < 0.7 && normalizedName(line["text"] as! String) == normalizedName(target)
    }
    guard let heading = heading else {
        return ["ok": false, "code": "target_mismatch", "error": "当前微信会话与指定联系人不匹配，已暂停读取正文。请在微信中打开「\(target)」。"]
    }
    let left = max(width * 0.20, CGFloat(heading["x"] as! Double) * width - 26)
    let top = headerHeight + 22
    let bottom = height * 0.77
    let rect = CGRect(x: left, y: top, width: width - left - 12, height: max(1, bottom - top))
    guard let bodyImage = image.cropping(to: rect) else {
        throw NSError(domain: "ChatLens", code: 2, userInfo: [NSLocalizedDescriptionKey: "无法裁剪聊天内容"])
    }
    let lines = try recognize(bodyImage)
    // OCR cannot reliably determine the speaker or distinguish text inside images.
    // Keep every role unknown; the user can correct it in the product UI.
    return ["ok": true, "header": target, "lines": lines,
            "captured_at": ISO8601DateFormatter().string(from: Date()),
            "scope": "visible-window", "speaker_warning": true]
}

final class AppDelegate: NSObject, NSApplicationDelegate, WKScriptMessageHandler, WKNavigationDelegate, NSWindowDelegate {
    var window: NSWindow!
    var web: WKWebView!
    var backend: Process?
    var clickObserver: Process?
    var backendOutput: Pipe?
    var captureBusy = false
    var requestURL: URL?
    var sessionToken = ""
    var bootBuffer = Data()
    var companion: NSPanel!
    var companionWeb: WKWebView!
    var followTimer: Timer?
    var panelVisibility = DecisionPanelVisibility()
    var lastGeometry = ""
    var snapshotSaved = false
    var clickAnchor: CGPoint?
    var clickWechatPID: pid_t?
    var selectionRequestBusy = false
    var lastKnownWechatWindow: (pid_t, CGRect)?
    var panelLayout = UserDefaults.standard.string(forKey: "decisionPanelLayout") ?? "docked"
    var layoutPreferences = UserDefaults.standard
    var expectedPanelFrame: NSRect?
    var dockSide = "right"
    var layoutHint = "拖动标题栏可自由放置"

    func applicationDidFinishLaunching(_ notification: Notification) {
        let config = WKWebViewConfiguration()
        config.websiteDataStore = .nonPersistent()
        config.userContentController.add(self, name: "native")
        web = WKWebView(frame: .zero, configuration: config)
        web.navigationDelegate = self
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1220, height: 830),
                          styleMask: [.titled, .closable, .miniaturizable, .resizable], backing: .buffered, defer: false)
        window.title = "点击消息 · 对话决策助手"
        window.minSize = NSSize(width: 960, height: 700)
        window.contentView = web
        window.delegate = self
        window.center()
        if !CommandLine.arguments.contains("--in-wechat") {
            window.makeKeyAndOrderFront(nil)
            NSApp.activate(ignoringOtherApps: true)
        }
        setupCompanion()
        setupMenu()
        startBackend()
    }

    func setupCompanion() {
        let configuration = WKWebViewConfiguration()
        configuration.websiteDataStore = .nonPersistent()
        configuration.userContentController.add(self, name: "native")
        companionWeb = WKWebView(frame: .zero, configuration: configuration)
        companionWeb.navigationDelegate = self
        companion = NSPanel(contentRect: NSRect(x: 0, y: 0, width: 350, height: 620),
                            styleMask: [.titled, .utilityWindow, .nonactivatingPanel, .resizable], backing: .buffered, defer: false)
        companion.title = "所选消息 · 决策建议"
        companion.level = .floating
        companion.isFloatingPanel = true
        companion.hidesOnDeactivate = false
        companion.isReleasedWhenClosed = false
        companion.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        companion.contentView = companionWeb
        companion.minSize = NSSize(width: 310, height: 360)
        companion.isMovableByWindowBackground = true
        companion.center()
        expectedPanelFrame = companion.frame
        companion.delegate = self
        if panelLayout == "floating", let saved = layoutPreferences.string(forKey: "decisionPanelFrame") {
            let frame = NSRectFromString(saved)
            if frame.width >= 310 && frame.height >= 360 && NSScreen.screens.contains(where: { $0.visibleFrame.intersection(frame).width >= 100 && $0.visibleFrame.intersection(frame).height >= 100 }) {
                setPanelFrame(frame)
            } else { panelLayout = "docked" }
        }
        followTimer = Timer.scheduledTimer(withTimeInterval: 0.8, repeats: true) { [weak self] _ in
            self?.ensureClickObserver();self?.readClickSelection();self?.followWechat()
        }
        followWechat()
    }

    func followWechat() {
        let front = NSWorkspace.shared.frontmostApplication
        let applications = NSWorkspace.shared.runningApplications.filter {
            $0.bundleIdentifier == "com.tencent.xinWeChat" || $0.bundleIdentifier == "io.github.timfang4162.wechat4-macos-frida-export.sip"
        }
        let pids = Set(applications.map { $0.processIdentifier })
        let foregroundWechat = front.map { pids.contains($0.processIdentifier) } ?? false
        let list = CGWindowListCopyWindowInfo([.optionOnScreenOnly, .excludeDesktopElements], kCGNullWindowID) as? [[String: Any]] ?? []
        let matches = list.compactMap { item -> (pid_t, CGRect)? in
            guard let pid = item[kCGWindowOwnerPID as String] as? Int,
                  pids.contains(pid_t(pid)), (item[kCGWindowLayer as String] as? Int) == 0,
                  let box = item[kCGWindowBounds as String] as? [String: Any],
                  let rect = CGRect(dictionaryRepresentation: box as CFDictionary),
                  rect.width > 650, rect.height > 450 else { return nil }
            if foregroundWechat && pid_t(pid) != front?.processIdentifier { return nil }
            return (pid_t(pid), rect)
        }
        let clickedWindow = matches.first { pid, rect in
            clickWechatPID == pid && clickAnchor.map { rect.contains($0) } == true
        }
        let preferredWindow = matches.first { $0.0 == lastKnownWechatWindow?.0 }
        let observed = clickedWindow ?? preferredWindow ?? matches.max { $0.1.width * $0.1.height < $1.1.width * $1.1.height }
        let chosen = retainWechatWindow(observed: observed, previous: lastKnownWechatWindow)
        lastKnownWechatWindow = chosen
        let visible = panelVisibility.visible
        let probe: [String: Any] = ["front_bundle": front?.bundleIdentifier ?? "", "wechat_app_count": applications.count,
            "eligible_windows": matches.count, "visible": visible,
            "selection_latched": panelVisibility.hasSelectedMessage, "user_enabled": panelVisibility.enabled,
            "using_cached_bounds": observed == nil && chosen != nil, "checked_at": Date().timeIntervalSince1970]
        try? jsonString(probe).write(to: diagnosticsRoot.appendingPathComponent("window-probe.json"), atomically: true, encoding: .utf8)
        if let (pid, bounds) = chosen {
            let desktopTop = NSScreen.screens.first?.frame.maxY ?? 0
            let wechatFrame = NSRect(x: bounds.minX, y: desktopTop-bounds.maxY, width: bounds.width, height: bounds.height)
            let screen = NSScreen.screens.max { $0.frame.intersection(wechatFrame).area < $1.frame.intersection(wechatFrame).area } ?? NSScreen.main
            if panelLayout == "docked", let available = screen?.visibleFrame {
                let placement = dockPlacement(wechat:wechatFrame, available:available)
                dockSide = placement.side
                if dockSide == "right" {
                    layoutHint = "贴靠微信右侧 · 拖动标题栏可自由放置"
                } else if dockSide == "left" {
                    layoutHint = "右侧空间不足，已贴靠左侧"
                } else {
                    panelLayout = "floating"; dockSide = "none"
                    layoutHint = "两侧空间不足，请拖动标题栏调整位置"
                    layoutPreferences.set(panelLayout, forKey: "decisionPanelLayout")
                }
                setPanelFrame(placement.frame)
                if panelLayout == "floating" { saveFloatingFrame() }
            }
            let frame = companion.frame
            let topY = desktopTop-frame.maxY
                let metadata: [String: Any] = ["attached": true, "visible": visible, "wechat_pid": Int(pid), "width": frame.width,
                    "height": frame.height, "presentation": "draggable_decision_panel", "layout":panelLayout, "dock_side":dockSide,
                    "overlaps_wechat":frame.intersects(wechatFrame), "manual_position_retained":panelLayout == "floating",
                    "screen_image_used_for_position": false, "analysis_text_source": "database",
                    "frame_x":frame.minX,"frame_top_y":topY,"click_x":clickAnchor?.x ?? -1,"click_y":clickAnchor?.y ?? -1,
                    "accessibility_permission_requested": false,
                    "official_app_modified": false, "using_cached_bounds": observed == nil,
                    "checked_at": Date().timeIntervalSince1970]
                try? jsonString(metadata).write(to: diagnosticsRoot.appendingPathComponent("window-attachment.json"), atomically: true, encoding: .utf8)
        }
        // The floating level keeps the panel above the chat without repeatedly
        // ordering it forward every timer tick or requiring WeChat focus.
        if visible && !companion.isVisible { companion.orderFrontRegardless() }
        else if !visible && companion.isVisible { companion.orderOut(nil) }
        companionWeb.evaluateJavaScript("window.chatlensSetVisible && window.chatlensSetVisible(\(visible ? "true" : "false"))", completionHandler: nil)
        let layout: [String:Any] = ["mode":panelLayout,"side":dockSide,"hint":layoutHint]
        companionWeb.evaluateJavaScript("window.chatlensLayout && window.chatlensLayout(\(jsonString(layout)))", completionHandler:nil)
        web.evaluateJavaScript("window.chatlensLayout && window.chatlensLayout(\(jsonString(layout)))", completionHandler:nil)
        web.evaluateJavaScript("window.chatlensSetVisible && window.chatlensSetVisible(\(window.isVisible && !window.isMiniaturized ? "true" : "false"))", completionHandler: nil)
        if requestURL != nil {
            let check = "({cards:document.querySelectorAll('.decision-card').length,ranked_actions:document.querySelectorAll('.decision-card:not(.partial) .action').length,navigation_tabs:document.querySelectorAll('nav button').length,has_error:Boolean(document.getElementById('error')&&!document.getElementById('error').hidden),consent_visible:Boolean(document.getElementById('consent')&&!document.getElementById('consent').hidden),consent_enabled:Boolean(document.getElementById('consent-enable')&&!document.getElementById('consent-enable').disabled),contacts_count:document.querySelectorAll('#contact-select option').length,contact_groups:document.querySelectorAll('#contact-select optgroup').length,horizontal_overflow:document.documentElement.scrollWidth>document.documentElement.clientWidth,emotion_count:document.querySelectorAll('.emotion-chip').length,selected_quote_visible:Boolean(document.getElementById('selected-quote')&&!document.getElementById('selected-quote').hidden),selected_analysis_enabled:Boolean(document.getElementById('analyze-selected')&&!document.getElementById('analyze-selected').disabled)})"
            companionWeb.evaluateJavaScript(check) { [weak self] value, error in
                guard let self = self, error == nil, var metadata = value as? [String: Any] else { return }
                metadata["checked_at"] = Date().timeIntervalSince1970
                try? jsonString(metadata).write(to: diagnosticsRoot.appendingPathComponent("companion-ui-check.json"), atomically: true, encoding: .utf8)
                if !self.snapshotSaved && (metadata["ranked_actions"] as? Int ?? 0) >= 2 {
                    self.snapshotSaved = true
                    self.companionWeb.takeSnapshot(with: nil) { image, _ in
                        guard let data = image?.tiffRepresentation, let rep = NSBitmapImageRep(data: data),
                              let png = rep.representation(using: .png, properties: [:]) else { return }
                        let folder = screenshotRoot
                        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
                        let file = folder.appendingPathComponent("decision-card-live.png")
                        try? png.write(to: file, options: .atomic)
                        try? FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: file.path)
                    }
                }
            }
        }
    }

    func readClickSelection() {
        guard let base=requestURL, !selectionRequestBusy else { return }
        selectionRequestBusy=true
        var request=URLRequest(url:base.appendingPathComponent("api/status"));request.timeoutInterval=2
        request.setValue(sessionToken,forHTTPHeaderField:"X-ChatLens-Token")
        URLSession.shared.dataTask(with:request){[weak self] data,_,_ in
            DispatchQueue.main.async {
                guard let self=self else{return};self.selectionRequestBusy=false
                guard let data=data,let value=(try? JSONSerialization.jsonObject(with:data)) as? [String:Any] else{return}
                self.panelVisibility.receiveContact(value)
                self.panelVisibility.receive(value["selection"] as? [String:Any])
                if let selection=value["selection"] as? [String:Any],let location=selection["location"] as? [String:Any],
                   let x=location["x"] as? Double,let y=location["y"] as? Double,let pid=location["wechat_pid"] as? Int {
                    self.clickAnchor=CGPoint(x:x,y:y);self.clickWechatPID=pid_t(pid)
                }
                self.followWechat()
            }
        }.resume()
    }

    @objc func showMainWindow() {
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    @objc func toggleOverlay() { panelVisibility.toggle(); followWechat() }
    func setPanelFrame(_ frame: NSRect) {
        expectedPanelFrame = frame
        if !NSEqualRects(companion.frame, frame) { companion.setFrame(frame, display: true) }
    }
    func saveFloatingFrame() {
        layoutPreferences.set("floating", forKey:"decisionPanelLayout")
        layoutPreferences.set(NSStringFromRect(companion.frame), forKey:"decisionPanelFrame")
    }
    func panelWasMoved(_ notification: Notification) {
        guard let moved = notification.object as? NSWindow, moved === companion,
              expectedPanelFrame.map({ !NSEqualRects($0,companion.frame) }) ?? true else { return }
        panelLayout = "floating"; dockSide = "none"; layoutHint = "自由窗口 · 位置已保留"
        expectedPanelFrame = companion.frame
        saveFloatingFrame()
    }
    func windowDidMove(_ notification: Notification) { panelWasMoved(notification) }
    func windowDidResize(_ notification: Notification) { panelWasMoved(notification) }
    func windowShouldClose(_ sender: NSWindow) -> Bool {
        if sender === companion { panelVisibility.dismiss(); companion.orderOut(nil); return false }
        return true
    }
    @objc func dockPanel() {
        panelLayout="docked"; layoutPreferences.set(panelLayout,forKey:"decisionPanelLayout")
        followWechat()
    }
    @objc func floatPanel() {
        panelLayout="floating"; dockSide="none"; layoutHint="自由窗口 · 拖动标题栏调整位置"
        saveFloatingFrame(); followWechat()
    }

    func setupMenu() {
        let menu = NSMenu()
        let appItem = NSMenuItem()
        menu.addItem(appItem)
        let appMenu = NSMenu()
        appMenu.addItem(withTitle: "打开决策卡窗口", action: #selector(showMainWindow), keyEquivalent: "o")
        appMenu.addItem(withTitle: "显示 / 隐藏微信浮层", action: #selector(toggleOverlay), keyEquivalent: "j")
        appMenu.addItem(withTitle: "卡片贴靠微信右侧", action: #selector(dockPanel), keyEquivalent: "d")
        appMenu.addItem(withTitle: "卡片自由拖动", action: #selector(floatPanel), keyEquivalent: "f")
        appMenu.addItem(withTitle: "退出微信对话助手", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        appItem.submenu = appMenu
        let editItem = NSMenuItem()
        menu.addItem(editItem)
        let editMenu = NSMenu(title: "编辑")
        editMenu.addItem(withTitle: "复制", action: #selector(NSText.copy(_:)), keyEquivalent: "c")
        editMenu.addItem(withTitle: "粘贴", action: #selector(NSText.paste(_:)), keyEquivalent: "v")
        editMenu.addItem(withTitle: "全选", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a")
        editItem.submenu = editMenu
        NSApp.mainMenu = menu
    }

    func startBackend() {
        // A prestarted loopback service avoids launching Python under a newly
        // signed GUI identity when no additional macOS permissions are available.
        let existingSession = ProcessInfo.processInfo.environment
        if let portText = existingSession["CHATLENS_PORT"], let port = Int(portText),
           let token = existingSession["CHATLENS_TOKEN"], token.count == 64,
           token.range(of: "^[0-9a-f]{64}$", options: .regularExpression) != nil {
            connectBackend(port: port, token: token)
            return
        }
        if let data = try? Data(contentsOf: supportRoot.appendingPathComponent("bridge-session.json")),
           let value = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let port = value["port"] as? Int, let token = value["token"] as? String,
           token.range(of: "^[0-9a-f]{64}$", options: .regularExpression) != nil {
            connectBackend(port: port, token: token)
            return
        }
        let process = Process()
        let pipe = Pipe()
        backend = process
        backendOutput = pipe
        process.executableURL = pythonExecutable
        process.arguments = ["-m", "chatlens", "serve"]
        process.currentDirectoryURL = projectRoot
        var env = ProcessInfo.processInfo.environment
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["HF_HUB_DISABLE_TELEMETRY"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        env["CHATLENS_DATA_DIR"] = supportRoot.path
        env["PYTHONPATH"] = projectRoot.appendingPathComponent("src").path + (env["PYTHONPATH"].map { ":" + $0 } ?? "")
        process.environment = env
        process.standardOutput = pipe
        // Framework warnings contain no conversation content. No chat logging.
        process.standardError = FileHandle.nullDevice
        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            if data.isEmpty { return }
            DispatchQueue.main.async {
                guard let self = self, self.requestURL == nil else { return }
                self.bootBuffer.append(data)
                guard let line = String(data: self.bootBuffer, encoding: .utf8)?.split(separator: "\n").first,
                      let jsonData = String(line).data(using: .utf8),
                      let response = try? JSONSerialization.jsonObject(with: jsonData) as? [String: Any],
                      let port = response["port"] as? Int, let token = response["token"] as? String else { return }
                self.connectBackend(port: port, token: token)
            }
        }
        process.terminationHandler = { [weak self] _ in
            DispatchQueue.main.async {
                self?.deliver(["ok": false, "error": "本地模型服务已退出，请重新打开应用。"], id: "service")
            }
        }
        do { try process.run() }
        catch {
            web.loadHTMLString("<h2>无法启动本地模型服务</h2><p>请检查项目目录内的 Python 环境。</p>", baseURL: nil)
        }
    }

    func connectBackend(port: Int, token: String) {
        sessionToken = token
        requestURL = URL(string: "http://127.0.0.1:\(port)/")!
        var request = URLRequest(url: requestURL!)
        request.setValue(token, forHTTPHeaderField: "X-ChatLens-Token")
        web.load(request)
        var panelRequest = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/plugin")!)
        panelRequest.setValue(token, forHTTPHeaderField: "X-ChatLens-Token")
        companionWeb.load(panelRequest)
        ensureClickObserver()
    }

    func ensureClickObserver() {
        guard requestURL != nil, sessionToken.count == 64 else { return }
        if clickObserver?.isRunning == true { return }
        let owned = buildRoot.appendingPathComponent("click-observer-process.json")
        if let data = try? Data(contentsOf: owned),
           let info = (try? JSONSerialization.jsonObject(with: data)) as? [String:Any],
           let pid = info["pid"] as? Int, kill(pid_t(pid), 0) == 0 { return }
        let binary = buildRoot.appendingPathComponent("WeChatClickObserver")
        guard FileManager.default.isExecutableFile(atPath: binary.path) else { return }
        let process = Process()
        process.executableURL = binary
        var environment = ProcessInfo.processInfo.environment
        environment["CHATLENS_PORT"] = String(requestURL?.port ?? 0)
        environment["CHATLENS_TOKEN"] = sessionToken
        environment["CHATLENS_DIAGNOSTICS"] = diagnosticsRoot.path
        environment["CHATLENS_PRIVATE"] = supportRoot.path
        environment["CHATLENS_ALL_CONTACTS"] = "1"
        process.environment = environment
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        do {
            try process.run()
            clickObserver = process
            try? jsonString(["pid":Int(process.processIdentifier),"owner":"chatlens-native"])
                .write(to:owned,atomically:true,encoding:.utf8)
        } catch { /* No login, permission dialog, or external fallback. */ }
    }

    func deliver(_ result: [String: Any], id: String) {
        let payload: [String: Any] = ["id": id, "result": result]
        web.evaluateJavaScript("window.receiveNative && window.receiveNative(\(jsonString(payload)))", completionHandler: nil)
    }

    func userContentController(_ userContentController: WKUserContentController, didReceive message: WKScriptMessage) {
        guard message.frameInfo.isMainFrame, message.frameInfo.securityOrigin.host == "127.0.0.1",
              let body = message.body as? [String: Any], let action = body["action"] as? String,
              let id = body["id"] as? String else { return }
        if action == "copy", let text = body["text"] as? String {
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(text, forType: .string)
            deliver(["ok": true], id: id)
        } else if action == "open-history" {
            showMainWindow()
        } else if action == "hide-card" {
            panelVisibility.dismiss();companion.orderOut(nil)
        } else if action == "layout-dock" {
            dockPanel()
        } else if action == "layout-float" {
            floatPanel()
        } else if action == "permission" {
            let granted = CGRequestScreenCaptureAccess()
            if !granted {
                NSWorkspace.shared.open(URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture")!)
            }
            deliver(["ok": granted, "code": "permission", "error": granted ? "权限已可用" : "请在系统设置中开启本应用的屏幕录制权限，然后重新打开应用。"], id: id)
        } else if action == "capture", let target = body["target"] as? String, !target.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            if captureBusy { deliver(["ok": false, "code": "busy", "error": "正在读取微信窗口"], id: id); return }
            captureBusy = true
            Task { @MainActor in
                do {
                    let result = try await captureWechat(target: target)
                    deliver(result, id: id)
                } catch {
                    deliver(["ok": false, "code": "capture_error", "error": "读取失败：\(error.localizedDescription)"], id: id)
                }
                captureBusy = false
            }
        }
    }

    func webView(_ webView: WKWebView, decidePolicyFor navigationAction: WKNavigationAction,
                 decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
        guard let url = navigationAction.request.url else { decisionHandler(.cancel); return }
        decisionHandler(url.scheme == "http" && url.host == "127.0.0.1" && url.port == requestURL?.port ? .allow : .cancel)
    }

    // This is a resident chat companion; closing a view must not stop clicking.
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { false }
    func applicationWillTerminate(_ notification: Notification) {
        backendOutput?.fileHandleForReading.readabilityHandler = nil
        followTimer?.invalidate()
        let owned=buildRoot.appendingPathComponent("click-observer-process.json")
        if let data=try? Data(contentsOf:owned),let meta=(try? JSONSerialization.jsonObject(with:data)) as? [String:Any],let pid=meta["pid"] as? Int {
            let check=Process();let output=Pipe();check.executableURL=URL(fileURLWithPath:"/bin/ps")
            check.arguments=["-p",String(pid),"-o","args="];check.standardOutput=output;check.standardError=FileHandle.nullDevice
            if (try? check.run()) != nil {
                check.waitUntilExit();let args=String(data:output.fileHandleForReading.readDataToEndOfFile(),encoding:.utf8) ?? ""
                if args.contains(buildRoot.appendingPathComponent("WeChatClickObserver").path){kill(pid_t(pid),SIGTERM)}
            }
        }
        if backend?.isRunning == true { backend?.terminate() }
    }
}

if CommandLine.arguments.count == 4 && CommandLine.arguments[1] == "--ocr-fixture" {
    let url = URL(fileURLWithPath: CommandLine.arguments[2])
    if let image = NSImage(contentsOf: url), let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) {
        do { print(jsonString(try analyzeWechatImage(cgImage, target: CommandLine.arguments[3]))) }
        catch { print(jsonString(["ok": false, "error": error.localizedDescription])); exit(1) }
    } else { exit(2) }
    exit(0)
}

if CommandLine.arguments.contains("--stability-self-test") {
    func selection(_ revision: String, at time: TimeInterval, status: String = "matched") -> [String: Any] {
        ["status":status,"message":["id":"synthetic-message"],"revision":revision,"selected_at":time]
    }
    var state=DecisionPanelVisibility()
    var checks=[!state.visible]
    state.receive(selection("first",at:10));checks.append(state.visible)
    state.receive(nil);checks.append(state.visible)
    state.receive(selection("empty-click",at:11,status:"no_match"));checks.append(state.visible && state.revision=="first")
    state.receive(selection("ambiguous-click",at:12,status:"ambiguous"));checks.append(state.visible && state.revision=="first")
    state.dismiss(at:15);checks.append(!state.visible)
    state.receive(selection("first",at:10));checks.append(!state.visible)
    state.receive(selection("delayed",at:14));checks.append(!state.visible)
    state.receive(selection("later",at:16));checks.append(state.visible)
    state.dismiss(at:17)
    state.receive(selection("later",at:16));checks.append(!state.visible)
    state.receive(selection("latest",at:18));checks.append(state.visible)
    state.receive(["status":"matched","revision":"missing-message","selected_at":19]);checks.append(state.visible && state.revision=="latest")
    state.receive(selection("own-message",at:20,status:"self_message"));checks.append(state.visible && state.revision=="own-message")
    state.receive(selection("unknown-speaker",at:21,status:"unknown_role"));checks.append(state.visible && state.revision=="unknown-speaker")
    let original:(pid_t,CGRect)=(100,CGRect(x:100,y:80,width:900,height:650))
    let retained=retainWechatWindow(observed:nil,previous:original)
    checks.append(retained?.0==original.0 && retained?.1==original.1)
    let moved:(pid_t,CGRect)=(100,CGRect(x:150,y:100,width:900,height:650))
    let updated=retainWechatWindow(observed:moved,previous:original)
    checks.append(updated?.1==moved.1)
    checks.append(retainWechatWindow(observed:nil,previous:nil)==nil)
    var consent = DecisionPanelVisibility()
    consent.receiveContact(["contact_revision":"new-chat", "contact_observed_at":20.0,
                            "display_name":"合成联系人", "requires_consent":true])
    checks.append(consent.visible && consent.revision.isEmpty)
    consent.dismiss(at:21)
    consent.receiveContact(["contact_revision":"new-chat", "contact_observed_at":20.0,
                            "display_name":"合成联系人", "requires_consent":true])
    checks.append(!consent.visible)
    consent.receiveContact(["contact_revision":"late-chat", "contact_observed_at":20.5,
                            "display_name":"合成联系人", "requires_consent":true])
    checks.append(!consent.visible)
    consent.receiveContact(["contact_revision":"later-chat", "contact_observed_at":22.0,
                            "display_name":"另一联系人", "requires_consent":true])
    checks.append(consent.visible)
    consent.dismiss(at:23)
    consent.receiveContact(["contact_revision":"unobserved-default", "contact_observed_at":0.0,
                            "display_name":"合成联系人"])
    checks.append(!consent.visible)
    print(jsonString(["stability_self_test":checks.allSatisfy{$0},"checks":checks.count,"passed":checks.filter{$0}.count,
        "selection_latched":true,"dismissal_preserved":true,"cached_bounds_retained":true,"synthetic_only":true]))
    exit(checks.allSatisfy{$0} ? 0 : 1)
}

if CommandLine.arguments.contains("--layout-self-test") {
    let screen=NSRect(x:0,y:90,width:1470,height:833)
    let chat=NSRect(x:131,y:102,width:880,height:641)
    let right=dockPlacement(wechat:chat,available:screen)
    precondition(right.side=="right" && !right.frame.intersects(chat) && screen.contains(right.frame))
    let nearRight=NSRect(x:520,y:102,width:880,height:641)
    let left=dockPlacement(wechat:nearRight,available:screen)
    precondition(left.side=="left" && !left.frame.intersects(nearRight) && screen.contains(left.frame))
    let wide=NSRect(x:30,y:102,width:1400,height:641)
    let fallback=dockPlacement(wechat:wide,available:screen)
    precondition(fallback.side=="none" && screen.contains(fallback.frame))
    _ = NSApplication.shared
    let test=AppDelegate(), suite="local.chatlens.layout-tests.\(ProcessInfo.processInfo.processIdentifier)"
    test.layoutPreferences=UserDefaults(suiteName:suite)!
    test.layoutPreferences.removePersistentDomain(forName:suite)
    test.panelLayout="docked"
    test.companion=NSPanel(contentRect:.zero,styleMask:[.titled,.resizable,.nonactivatingPanel],backing:.buffered,defer:false)
    test.setPanelFrame(right.frame)
    test.windowDidMove(Notification(name:NSWindow.didMoveNotification,object:test.companion))
    precondition(test.panelLayout=="docked", "Programmatic docking must not become a manual move")
    let dragged=NSRect(x:20,y:160,width:360,height:500)
    test.companion.setFrame(dragged,display:false)
    test.windowDidMove(Notification(name:NSWindow.didMoveNotification,object:test.companion))
    precondition(test.panelLayout=="floating" && test.layoutPreferences.string(forKey:"decisionPanelLayout")=="floating")
    precondition(NSRectFromString(test.layoutPreferences.string(forKey:"decisionPanelFrame")!)==dragged)
    test.layoutPreferences.removePersistentDomain(forName:suite)
    print(jsonString(["passed":5,"right_dock_no_overlap":true,"left_fallback_no_overlap":true,
        "insufficient_space_fallback":true,"manual_move_saved":true,"programmatic_move_ignored":true,
        "physical_mouse_drag_tested":false]))
    exit(0)
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.regular)
app.run()
