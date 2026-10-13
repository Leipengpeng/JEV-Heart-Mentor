import Cocoa
import ApplicationServices
import ScreenCaptureKit
import Vision

let env = ProcessInfo.processInfo.environment
let port = Int(env["CHATLENS_PORT"] ?? "0") ?? 0
let token = env["CHATLENS_TOKEN"] ?? ""
let defaultTargets: [String] = []
// Legacy OCR aliases are supplemented by the local contact directory. AX
// discovery is not limited to this list; analysis consent is enforced by API.
let targets: [String] = {
    guard let raw = env["CHATLENS_TARGETS"] else { return defaultTargets }
    guard let data = raw.data(using: .utf8),
          let names = (try? JSONSerialization.jsonObject(with: data)) as? [String] else { return [] }
    let clean = names.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
    guard !clean.isEmpty, clean.allSatisfy({ !$0.isEmpty && $0.count <= 100 }) else { return [] }
    return Array(Set(clean)).sorted()
}()
let diagnosticRoot = URL(fileURLWithPath: env["CHATLENS_DIAGNOSTICS"] ?? "")
let privateRoot = URL(fileURLWithPath: env["CHATLENS_PRIVATE"] ?? "")
let wechatBundles = Set(["com.tencent.xinWeChat", "io.github.timfang4162.wechat4-macos-frida-export.sip"])

func json(_ value: Any) -> Data { (try? JSONSerialization.data(withJSONObject: value)) ?? Data() }
func record(_ value: [String: Any]) {
    let file = diagnosticRoot.appendingPathComponent("click-observer.json")
    try? json(value).write(to: file, options: .atomic)
    try? FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: file.path)
}
func api(_ path: String, _ body: [String: Any], timeout: TimeInterval = 8) async -> [String: Any]? {
    guard port > 0, token.count == 64, let url = URL(string: "http://127.0.0.1:\(port)\(path)") else { return nil }
    var r = URLRequest(url: url); r.httpMethod = "POST"; r.timeoutInterval = timeout
    r.setValue(token, forHTTPHeaderField: "X-ChatLens-Token")
    r.setValue("application/json", forHTTPHeaderField: "Content-Type")
    r.httpBody = json(body)
    guard let (data, response) = try? await URLSession.shared.data(for: r),
          (response as? HTTPURLResponse)?.statusCode == 200 else { return nil }
    return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
}
// AX supplies UI location only. Canonical identity and message IDs are resolved
// against the local database by the service; no chat content is logged here.
func axAttribute(_ element: AXUIElement, _ attribute: String) -> CFTypeRef? {
    var value: CFTypeRef?
    guard AXUIElementCopyAttributeValue(element, attribute as CFString, &value) == .success else { return nil }
    return value
}
func axElement(_ value: CFTypeRef?) -> AXUIElement? {
    guard let value=value, CFGetTypeID(value)==AXUIElementGetTypeID() else {return nil}
    return unsafeBitCast(value,to:AXUIElement.self)
}
func axRect(_ element: AXUIElement) -> CGRect? {
    guard let p=axAttribute(element,kAXPositionAttribute),let s=axAttribute(element,kAXSizeAttribute),
          CFGetTypeID(p)==AXValueGetTypeID(),CFGetTypeID(s)==AXValueGetTypeID() else{return nil}
    var point=CGPoint.zero;var size=CGSize.zero
    guard AXValueGetValue(unsafeBitCast(p,to:AXValue.self),.cgPoint,&point),
          AXValueGetValue(unsafeBitCast(s,to:AXValue.self),.cgSize,&size),size.width>0,size.height>0 else{return nil}
    return CGRect(origin:point,size:size)
}
func axChildren(_ element: AXUIElement) -> [AXUIElement] {
    // Some virtualized containers expose rendered rows only through
    // AXVisibleChildren/AXContents. Try those only when AXChildren is empty.
    for attribute in [kAXChildrenAttribute,"AXVisibleChildren","AXContents"] {
        if let raw=axAttribute(element,attribute) as? [AnyObject] {
            let children=raw.prefix(180).compactMap{axElement($0)}
            if !children.isEmpty{return children}
        }
    }
    return []
}
struct AXNode {
    let element: AXUIElement
    let identifier: String
    let role: String
    let text: String
    let rect: CGRect?
}
func axNode(_ element: AXUIElement) -> AXNode {
    let attributes=[kAXIdentifierAttribute,kAXRoleAttribute,kAXTitleAttribute,kAXValueAttribute,kAXPositionAttribute,kAXSizeAttribute]
    var values: CFArray?
    let error=AXUIElementCopyMultipleAttributeValues(element,attributes as CFArray,AXCopyMultipleAttributeOptions(rawValue:0),&values)
    let items=(error == .success ? values as? [Any] : nil) ?? []
    func string(_ index:Int)->String {items.count>index ? (items[index] as? String ?? "") : ""}
    var rect:CGRect?
    if items.count>=6,CFGetTypeID(items[4] as CFTypeRef)==AXValueGetTypeID(),CFGetTypeID(items[5] as CFTypeRef)==AXValueGetTypeID(){
        var p=CGPoint.zero;var s=CGSize.zero
        if AXValueGetValue(unsafeBitCast(items[4] as CFTypeRef,to:AXValue.self),.cgPoint,&p),
           AXValueGetValue(unsafeBitCast(items[5] as CFTypeRef,to:AXValue.self),.cgSize,&s),s.width>0,s.height>0{rect=CGRect(origin:p,size:s)}
    }
    let title=string(2).trimmingCharacters(in:.whitespacesAndNewlines)
    let value=string(3).trimmingCharacters(in:.whitespacesAndNewlines)
    // The chat title's text is exposed in AXValue on some WeChat 4.x builds.
    let id=string(0)
    return AXNode(element:element,identifier:id,role:string(1),text:id=="big_title_line_h_view" && !value.isEmpty ? value : (!title.isEmpty ? title : value),rect:rect)
}
struct AXMessage {
    let element: AXUIElement
    let text: String
    let rect: CGRect
}
struct AXSnapshot {
    let title: String?
    let windowFrame: CGRect?
    let messageFrame: CGRect?
    let messages: [AXMessage]
    let scanned: Int
    let bubbleCount: Int
    let virtualCount: Int
    let titleCount: Int
    let messageListCount: Int
    let elapsed: Double
    let roleCounts: [String:Int]
    let knownIdentifiers: [String:Int]
    var visibleTexts:[String]{messages.suffix(18).map{String($0.text.prefix(800))}}
}
func cleanChatTitle(_ text:String)->String? {
    let clean=text.trimmingCharacters(in:.whitespacesAndNewlines)
    guard !clean.isEmpty,clean.count<=120,!clean.contains("\n") else{return nil}
    return clean
}
func validAXMessage(_ text:String)->Bool {
    guard !text.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty else{return false}
    // Unlike OCR, a bubble's identifier establishes it as a message. Single
    // character replies and numeric messages must remain selectable.
    return text.count<=20000
}
func axSnapshot(pid:pid_t,includeMessages:Bool=true)->AXSnapshot {
    let began=Date();let app=AXUIElementCreateApplication(pid)
    AXUIElementSetMessagingTimeout(app,0.10)
    let window=axElement(axAttribute(app,kAXFocusedWindowAttribute)) ?? axElement(axAttribute(app,kAXMainWindowAttribute))
        ?? ((axAttribute(app,kAXWindowsAttribute) as? [AnyObject])?.compactMap{axElement($0)}.first)
    let root=window ?? app
    let frame=window.flatMap{axRect($0)}
    var titleNodes:[AXNode]=[];var lists:[AXNode]=[];var headerNodes:[AXNode]=[]
    var stack:[(AXUIElement,Int)]=[(root,0)];var seen=Set<CFHashCode>();var scanned=0
    var roles:[String:Int]=[:];var identifiers:[String:Int]=[:]
    // Read only a bounded window tree. Skip message list descendants here;
    // visible rows are read directly after selecting the actual conversation.
    while let (element,depth)=stack.popLast(),scanned<650,Date().timeIntervalSince(began)<0.55 {
        let key=CFHash(element);guard !seen.contains(key),depth<=24 else{continue};seen.insert(key)
        scanned+=1;let node=axNode(element)
        roles[node.role.isEmpty ? "unavailable" : node.role,default:0]+=1
        if ["big_title_line_h_view","chat_message_list","chat_bubble_item_view","virtual_cell","session_list"].contains(node.identifier){identifiers[node.identifier,default:0]+=1}
        else if node.identifier.hasPrefix("session_item_"){identifiers["session_item_prefix",default:0]+=1}
        if node.identifier=="big_title_line_h_view"{titleNodes.append(node)}
        if node.identifier=="chat_message_list" || (node.role==kAXListRole && ["Messages","消息"].contains(node.text)){
            lists.append(node);continue
        }
        if node.role==kAXStaticTextRole,let r=node.rect,let w=frame,r.minX>w.minX+w.width*0.22,r.minY<w.minY+w.height*0.13,
           node.identifier.hasPrefix("session_item_")==false,cleanChatTitle(node.text) != nil{headerNodes.append(node)}
        for child in axChildren(element).reversed(){stack.append((child,depth+1))}
    }
    // A title alone may survive temporarily while a conversation is closing.
    // Never observe it as the active chat without its visible message list.
    let list=lists.filter{$0.rect != nil}.max{($0.rect?.height ?? 0)<($1.rect?.height ?? 0)}
    var title:String?
    if let list=list,let listRect=list.rect,listRect.height>80,listRect.width>120 {
        let exact=titleNodes.compactMap{node -> String? in
            guard let r=node.rect,r.maxY<=listRect.minY+30,r.maxX>listRect.minX else{return nil}
            return cleanChatTitle(node.text)
        }
        let unique=Set(exact)
        if unique.count==1{title=unique.first}
        else if unique.isEmpty {
            // Identifier-free fallback is restricted to the top chat heading,
            // not arbitrary sidebar labels or bubble text.
            let eligible=headerNodes.filter{node in guard let r=node.rect else{return false};return r.maxY<=listRect.minY+4 && r.minX>=listRect.minX-20}
            let headings=Set(eligible.compactMap{cleanChatTitle($0.text)})
            if headings.count==1{title=headings.first}
        }
    }
    var messages:[AXMessage]=[];var bubbles=0;var virtual=0
    if includeMessages,let list=list,let viewport=list.rect {
        var rows:[(AXUIElement,Int)]=axChildren(list.element).reversed().map{($0,0)}
        var rowSeen=Set<CFHashCode>();var rowCount=0
        while let (element,depth)=rows.popLast(),rowCount<260,Date().timeIntervalSince(began)<0.85 {
            let key=CFHash(element);guard !rowSeen.contains(key),depth<=5 else{continue};rowSeen.insert(key);rowCount+=1
            let node=axNode(element)
            if node.identifier=="virtual_cell"{virtual+=1;continue}
            if node.identifier=="chat_bubble_item_view"{
                bubbles+=1
                if let r=node.rect,r.intersects(viewport),validAXMessage(node.text){messages.append(AXMessage(element:element,text:node.text,rect:r))}
                continue
            }
            for child in axChildren(element).reversed(){rows.append((child,depth+1))}
        }
    }
    messages.sort{$0.rect.midY<$1.rect.midY}
    return AXSnapshot(title:title,windowFrame:frame,messageFrame:list?.rect,messages:messages,scanned:scanned,bubbleCount:bubbles,
                      virtualCount:virtual,titleCount:titleNodes.count,messageListCount:lists.count,elapsed:Date().timeIntervalSince(began),roleCounts:roles,knownIdentifiers:identifiers)
}
func axMessageIndex(_ point:CGPoint,snapshot:AXSnapshot)->Int? {
    guard let viewport=snapshot.messageFrame,viewport.contains(point) else{return nil}
    // First use AX hit testing and bubble ancestry; wrapper rows often span the
    // full pane, so avatar/gutter clicks must not become message selections.
    var hit:AXUIElement?
    let system=AXUIElementCreateSystemWide();AXUIElementSetMessagingTimeout(system,0.1)
    _=AXUIElementCopyElementAtPosition(system,Float(point.x),Float(point.y),&hit)
    if var element=hit {
        for _ in 0..<14 {
            let node=axNode(element)
            let lowered=node.identifier.lowercased()
            if node.role==kAXImageRole || lowered.contains("avatar") || lowered.contains("head_image"){return nil}
            if node.identifier=="chat_bubble_item_view" {
                if let i=snapshot.messages.firstIndex(where:{CFEqual($0.element,element)}){
                    let rect=snapshot.messages[i].rect
                    let fullRow=rect.width>viewport.width*0.88
                    if fullRow && (point.x<viewport.minX+54 || point.x>viewport.maxX-54){return nil}
                    return i
                }
                break
            }
            guard let parent=axElement(axAttribute(element,kAXParentAttribute)) else{break};element=parent
        }
    }
    // Bounded geometric fallback for a WeChat build that returns a parent list
    // for coordinate hit tests. No nearest-message guesses outside row bounds.
    return axGeometricMessageIndex(point:point,viewport:viewport,rects:snapshot.messages.map{$0.rect})
}
func axGeometricMessageIndex(point:CGPoint,viewport:CGRect,rects:[CGRect])->Int? {
    guard viewport.contains(point) else{return nil}
    let hits=rects.enumerated().filter{_,rect in
        guard rect.contains(point) else{return false}
        return rect.width<=viewport.width*0.88 || (point.x>=viewport.minX+54 && point.x<=viewport.maxX-54)
    }
    return hits.count==1 ? hits.first?.offset : nil
}
func axLocator(index:Int,messages:[AXMessage])->[String:Any]? {
    guard messages.indices.contains(index) else{return nil}
    return ["clicked_texts":[String(messages[index].text.prefix(800))],
            "before_texts":messages[max(0,index-4)..<index].map{String($0.text.prefix(800))},
            "after_texts":messages[(index+1)..<min(messages.count,index+5)].map{String($0.text.prefix(800))}]
}
struct Line {
    let text: String
    let rect: CGRect
}
func recognize(_ image: CGImage, offset: CGPoint = .zero) throws -> [Line] {
    let request = VNRecognizeTextRequest(); request.recognitionLevel = .accurate
    request.recognitionLanguages = ["zh-Hans", "zh-Hant", "en-US"]; request.usesLanguageCorrection = true
    try VNImageRequestHandler(cgImage: image).perform([request])
    return (request.results ?? []).compactMap { observation in
        guard let result = observation.topCandidates(1).first else { return nil }
        let b = observation.boundingBox
        return Line(text: result.string, rect: CGRect(x: offset.x + b.minX * CGFloat(image.width),
            y: offset.y + (1-b.maxY) * CGFloat(image.height), width: b.width * CGFloat(image.width),
            height: b.height * CGFloat(image.height)))
    }.sorted { $0.rect.midY < $1.rect.midY }
}
func normalize(_ text: String) -> String { text.replacingOccurrences(of: "\\s+", with: "", options: .regularExpression) }
func recognizedContact(_ headings: [Line], imageWidth: CGFloat, allowedTargets: [String] = targets) -> String? {
    let eligible = headings.filter { $0.rect.minX > imageWidth * 0.2 }.map { normalize($0.text) }
    let matches = Set(allowedTargets.filter { eligible.contains(normalize($0)) })
    // Exact title identity is required. Multiple different allowed titles are
    // ambiguous and must not mix one friend's text with another friend's DB.
    return matches.count == 1 ? matches.first : nil
}
func messageLine(_ line: Line) -> Bool {
    let t = normalize(line.text)
    if t.isEmpty{return false}
    if t.range(of:"^[0-9]+$",options:.regularExpression) != nil{return true}
    return t.range(of: "^(今天|昨天|星期[一二三四五六日天]|[0-9年月日/:：.\\- ]+)$", options: .regularExpression) == nil
}
struct CaptureWindow { let frame: CGRect; let pid: pid_t; let id: Int }
func boundedWindowImage(_ window: CaptureWindow) throws -> CGImage? {
    // ScreenCaptureKit can leave its async completion pending on this renderer.
    // The system's window-specific capture has a bounded process lifetime and
    // shares the existing screen permission. No desktop-wide image is captured.
    let folder=FileManager.default.temporaryDirectory.appendingPathComponent("chatlens-private-capture")
    try FileManager.default.createDirectory(at:folder,withIntermediateDirectories:true,attributes:[.posixPermissions:0o700])
    let file=folder.appendingPathComponent(UUID().uuidString+".png")
    defer {try? FileManager.default.removeItem(at:file)}
    func runCapture(_ arguments:[String]) throws -> CGImage? {
        try? FileManager.default.removeItem(at:file)
        let process=Process();process.executableURL=URL(fileURLWithPath:"/usr/sbin/screencapture")
        process.arguments=arguments+[file.path]
        process.standardOutput=FileHandle.nullDevice;process.standardError=FileHandle.nullDevice
        try process.run();let started=Date()
        while process.isRunning && Date().timeIntervalSince(started)<4 {Thread.sleep(forTimeInterval:0.025)}
        if process.isRunning {kill(process.processIdentifier,SIGKILL);process.waitUntilExit();return nil}
        guard process.terminationStatus==0 else{return nil}
        try? FileManager.default.setAttributes([.posixPermissions:0o600],ofItemAtPath:file.path)
        return NSImage(contentsOf:file)?.cgImage(forProposedRect:nil,context:nil,hints:nil)
    }
    if let image=try runCapture(["-x","-o","-l",String(window.id)]){return image}
    // Some WeChat compositor windows refuse capture by ID even though their
    // pixels are visible. Fall back to the exact chat-window rectangle only
    // while WeChat is frontmost and no other app window covers that rectangle.
    // This never captures the whole desktop or prompts for another permission.
    guard NSWorkspace.shared.frontmostApplication?.processIdentifier==window.pid else{return nil}
    let list=CGWindowListCopyWindowInfo([.optionOnScreenOnly,.excludeDesktopElements],kCGNullWindowID) as? [[String:Any]] ?? []
    guard let index=list.firstIndex(where:{($0[kCGWindowNumber as String] as? Int)==window.id}) else{return nil}
    for item in list[..<index] {
        guard (item[kCGWindowAlpha as String] as? Double ?? 1)>0,
              (item[kCGWindowOwnerPID as String] as? Int) != Int(window.pid),
              let box=item[kCGWindowBounds as String] as? [String:Any],
              let frame=CGRect(dictionaryRepresentation:box as CFDictionary) else{continue}
        if frame.intersects(window.frame){return nil}
    }
    let rectangle=[window.frame.minX,window.frame.minY,window.frame.width,window.frame.height].map{String(Int($0.rounded()))}.joined(separator:",")
    if CommandLine.arguments.contains("--test-click"){print("{\"capture_rectangle_fallback\":true}")}
    return try runCapture(["-x","-R",rectangle])
}
func capture(pid: pid_t, aliases: [String] = targets, includeMessages: Bool = true, axTitle: String? = nil) async throws -> (CaptureWindow, CGImage, [Line], CGFloat, String)? {
    if CommandLine.arguments.contains("--test-click") { print(String(data:json(["capture_screen_access":CGPreflightScreenCaptureAccess()]),encoding:.utf8)!) }
    guard CGPreflightScreenCaptureAccess() else { return nil }
    let list=CGWindowListCopyWindowInfo([.optionOnScreenOnly,.excludeDesktopElements],kCGNullWindowID) as? [[String:Any]] ?? []
    let windows=list.compactMap { item -> CaptureWindow? in
        guard item[kCGWindowOwnerPID as String] as? Int == Int(pid),item[kCGWindowLayer as String] as? Int == 0,
              let id=item[kCGWindowNumber as String] as? Int,
              let box=item[kCGWindowBounds as String] as? [String:Any],let frame=CGRect(dictionaryRepresentation:box as CFDictionary),
              frame.width>650,frame.height>450 else{return nil}
        return CaptureWindow(frame:frame,pid:pid,id:id)
    }
    if CommandLine.arguments.contains("--test-click") { print(String(data:json(["capture_eligible_windows":windows.count]),encoding:.utf8)!) }
    guard let window=windows.max(by:{$0.frame.width*$0.frame.height<$1.frame.width*$1.frame.height}) else{return nil}
    let captured=try await Task.detached(operation:{try boundedWindowImage(window)}).value
    if CommandLine.arguments.contains("--test-click") { print(String(data:json(["capture_image_loaded":captured != nil]),encoding:.utf8)!) }
    guard let image=captured else{return nil}
    let scale = CGFloat(image.width)/window.frame.width
    let headerHeight = min(CGFloat(image.height)*0.11, 140*scale)
    guard let header = image.cropping(to: CGRect(x: 0,y: 0,width: image.width,height: Int(headerHeight))) else { return nil }
    let headings = try recognize(header)
    // OCR is a click-only fallback. Unknown contacts are identified solely from
    // the narrow chat heading; their consent remains a backend decision.
    let eligible=headings.filter{line in
        let t=normalize(line.text)
        return line.rect.minX>CGFloat(image.width)*0.23 && line.rect.minY>18*scale && line.rect.midY<95*scale &&
            t.count<=120 && !["微信","WeChat","搜索","Search","聊天信息"].contains(t) &&
            t.range(of:"^[.·•…0-9:/：]+$",options:.regularExpression)==nil
    }
    let exact=recognizedContact(eligible,imageWidth:CGFloat(image.width),allowedTargets:aliases)
    let geometric=Set(eligible.compactMap{cleanChatTitle($0.text)})
    // A single unknown heading is observation only. The service must resolve
    // a canonical DB contact, or clear stale advice without running analysis.
    let title=axTitle ?? exact ?? (geometric.count==1 ? geometric.first : nil)
    if CommandLine.arguments.contains("--test-click") {
        print(String(data:json(["capture_header_lines":headings.count,"capture_eligible_headers":eligible.count,
            "capture_unique_headers":geometric.count,"capture_alias_match":exact != nil,"capture_title_found":title != nil,
            "capture_pixel_width":image.width,"capture_pixel_height":image.height]),encoding:.utf8)!)
    }
    guard let target=title else{return nil}
    if !includeMessages{return (window,image,[],scale,target)}
    let region = CGRect(x: CGFloat(image.width)*0.22, y: headerHeight+6,
        width: CGFloat(image.width)*0.78-12, height: CGFloat(image.height)*0.82-headerHeight-6)
    guard let body = image.cropping(to: region) else { return nil }
    let lines = try recognize(body, offset: region.origin).filter(messageLine)
    if CommandLine.arguments.contains("--test-click") { print(String(data:json(["capture_message_lines":lines.count]),encoding:.utf8)!) }
    return (window,image,lines,scale,target)
}
func isMessageRegion(_ point: CGPoint, in frame: CGRect) -> Bool {
    point.x > frame.minX+frame.width*0.22 && point.x < frame.maxX &&
        point.y > frame.minY+frame.height*0.11 && point.y < frame.minY+frame.height*0.82
}
func messageLocator(local: CGPoint, lines: [Line], scale: CGFloat) -> [String: Any]? {
    let distances = lines.enumerated().map { i,l -> (Int,CGFloat) in
        let dx = max(0,max(l.rect.minX-local.x,local.x-l.rect.maxX))
        let dy = max(0,max(l.rect.minY-local.y,local.y-l.rect.maxY))
        return (i,dx*dx+dy*dy)
    }
    guard let best = distances.min(by: { $0.1 < $1.1 }), best.1 <= pow(32*scale,2) else { return nil }
    var lower=best.0, upper=best.0
    while lower>0 && best.0-lower<3 && abs(lines[lower-1].rect.minX-lines[best.0].rect.minX)<12*scale && lines[lower].rect.midY-lines[lower-1].rect.midY<26*scale { lower-=1 }
    while upper+1<lines.count && upper-best.0<3 && abs(lines[upper+1].rect.minX-lines[best.0].rect.minX)<12*scale && lines[upper+1].rect.midY-lines[upper].rect.midY<26*scale { upper+=1 }
    let group=lines[lower...upper].map{$0.text}.joined()
    return ["clicked_texts":Array(Set([lines[best.0].text,group])).map{String($0.prefix(800))},
        "before_texts":lines[max(0,lower-4)..<lower].map{String($0.text.prefix(800))},
        "after_texts":lines[(upper+1)..<min(lines.count,upper+5)].map{String($0.text.prefix(800))}]
}
func locator(_ point: CGPoint, _ window: CaptureWindow, _ lines: [Line], _ scale: CGFloat, target: String) -> [String: Any]? {
    let local = CGPoint(x: (point.x-window.frame.minX)*scale, y: (point.y-window.frame.minY)*scale)
    guard var payload = messageLocator(local: local, lines: lines, scale: scale) else { return nil }
    payload["target"] = target
    payload["location"] = ["x":point.x,"y":point.y,"wechat_pid":Int(window.pid)]
    return payload
}

func apiContacts() async->[String] {
    guard port>0,token.count==64,let url=URL(string:"http://127.0.0.1:\(port)/api/contacts") else{return []}
    var request=URLRequest(url:url);request.timeoutInterval=20;request.setValue(token,forHTTPHeaderField:"X-ChatLens-Token")
    guard let (data,response)=try? await URLSession.shared.data(for:request),(response as? HTTPURLResponse)?.statusCode==200,
          let object=try? JSONSerialization.jsonObject(with:data) else{return []}
    let dictionary=object as? [String:Any]
    let rows=(dictionary?["contacts"] as? [[String:Any]]) ?? (object as? [[String:Any]]) ?? []
    var names=Set((dictionary?["title_aliases"] as? [String] ?? []).compactMap{cleanChatTitle($0)})
    for row in rows {
        for key in ["display_name","name","remark","nickname"] {if let s=row[key] as? String,let clean=cleanChatTitle(s){names.insert(clean)}}
        for s in (row["aliases"] as? [String] ?? []) {if let clean=cleanChatTitle(s){names.insert(clean)}}
    }
    return names.sorted()
}
@MainActor final class Observer {
    struct PendingClick {let point:CGPoint;let pid:pid_t;let messageRegion:Bool;let sequence:Int}
    var monitor: Any?
    var busy=false
    var pending:PendingClick?
    var pendingObserve=false
    var clickSequence=0
    var clicks=0
    var mouseUps=0
    var lastFront=""
    var aliases=targets
    var aliasesLoaded=Date.distantPast
    var lastObserveKey=""
    var lastObserveTime=Date.distantPast
    var lastStatus="starting"
    var lastSource="none"
    var lastAXElapsed:Double=0
    var lastHeaderOCR=Date.distantPast
    var headerOCRCount=0
    var axCapability="unchecked"
    var lastAXCounts:[String:Int]=[:]
    func start() {
        monitor=NSEvent.addGlobalMonitorForEvents(matching:.leftMouseUp){[weak self] event in
            guard let self=self else{return}
            self.mouseUps+=1;self.lastFront=NSWorkspace.shared.frontmostApplication?.bundleIdentifier ?? ""
            guard let front=NSWorkspace.shared.frontmostApplication,wechatBundles.contains(front.bundleIdentifier ?? "") else{return}
            let mouse=NSEvent.mouseLocation
            let point=event.cgEvent?.location ?? CGPoint(x:mouse.x,y:(NSScreen.screens.first?.frame.maxY ?? 0)-mouse.y)
            let windows=CGWindowListCopyWindowInfo([.optionOnScreenOnly,.excludeDesktopElements],kCGNullWindowID) as? [[String:Any]] ?? []
            let nativePIDs=Set(NSWorkspace.shared.runningApplications.filter{$0.bundleIdentifier=="org.chatlens.opensource"}.map{Int($0.processIdentifier)})
            // Only the frontmost window under the pointer consumes a click.
            // A helper window behind WeChat may overlap its bounds and must
            // not suppress an actual message click in the foreground chat.
            let hit=windows.first{item in
                guard let b=item[kCGWindowBounds as String] as? [String:Any],
                      let r=CGRect(dictionaryRepresentation:b as CFDictionary) else{return false}
                return r.contains(point) && (item[kCGWindowAlpha as String] as? Double ?? 1)>0
            }
            guard !nativePIDs.contains(hit?[kCGWindowOwnerPID as String] as? Int ?? -1) else{return}
            guard let top=windows.first(where:{item in
                guard let b=item[kCGWindowBounds as String] as? [String:Any],let rect=CGRect(dictionaryRepresentation:b as CFDictionary) else{return false}
                return (item[kCGWindowOwnerPID as String] as? Int)==Int(front.processIdentifier) && rect.contains(point) &&
                    (item[kCGWindowLayer as String] as? Int)==0 && (item[kCGWindowAlpha as String] as? Double ?? 1)>0
            }),let box=top[kCGWindowBounds as String] as? [String:Any],let rect=CGRect(dictionaryRepresentation:box as CFDictionary) else{return}
            self.clickSequence+=1
            let click=PendingClick(point:point,pid:front.processIdentifier,messageRegion:isMessageRegion(point,in:rect),sequence:self.clickSequence)
            // Sidebar/header clicks observe after WeChat finishes replacing its
            // chat view. They do not select the text that previously occupied it.
            if click.messageRegion{self.pending=click;self.next()}
            else {Task{@MainActor in
                try? await Task.sleep(nanoseconds:180_000_000)
                guard click.sequence==self.clickSequence else{return}
                self.pending=click;self.next()
            }}
        }
        publish()
        Timer.scheduledTimer(withTimeInterval:1,repeats:true){[weak self] _ in Task{@MainActor in
            guard let self=self,let front=NSWorkspace.shared.frontmostApplication,wechatBundles.contains(front.bundleIdentifier ?? "") else{return}
            self.pendingObserve=true;self.next()
        }}
        Timer.scheduledTimer(withTimeInterval:5,repeats:true){[weak self] _ in Task{@MainActor in self?.publish()}}
    }
    func publish() {
        let metadata:[String:Any]=["running":true,"mouse_monitor":monitor != nil,"accessibility":AXIsProcessTrusted(),
            "screen_capture":CGPreflightScreenCaptureAccess(),"clicks_received":clicks,"mouse_ups":mouseUps,"last_front_bundle":lastFront,
            "last_status":lastStatus,"locator_source":lastSource,"ax_elapsed_ms":Int(lastAXElapsed*1000),"ax_counts":lastAXCounts,"ax_capability":axCapability,"header_ocr_count":headerOCRCount,
            "checked_at":Date().timeIntervalSince1970,"window_image_saved":false,"capture_backend":"bounded_single_window_command"]
        record(metadata);Task{_ = await api("/api/click-observer/status",metadata)}
    }
    func observe(_ title:String,visible:[String],pid:pid_t,source:String,force:Bool) async->[String:Any]? {
        let key="\(pid):\(title)"
        // A health poll is not a new user choice. In particular it must not
        // undo a manual history selection or a same-name click disambiguation.
        if !force,key==lastObserveKey{return nil}
        let result=await api("/api/contact/observe",["title":title,"visible_texts":visible,"source":source,"wechat_pid":Int(pid)],timeout:20)
        if result != nil{lastObserveKey=key;lastObserveTime=Date()}
        return result
    }
    func next() {
        guard !busy else{return}
        let click=pending;pending=nil
        let shouldObserve=pendingObserve;pendingObserve=false
        guard click != nil || shouldObserve else{return}
        let pid:pid_t
        if let click=click{pid=click.pid}
        else {guard let front=NSWorkspace.shared.frontmostApplication,wechatBundles.contains(front.bundleIdentifier ?? "") else{return};pid=front.processIdentifier}
        busy=true
        Task {
            defer{busy=false;next()}
            // The synchronous AX traversal runs away from the UI/event monitor.
            // It has explicit size/depth/time limits and never requests trust.
            let snapshot=await Task.detached{axSnapshot(pid:pid,includeMessages:click != nil)}.value
            lastAXElapsed=snapshot.elapsed
            axCapability=snapshot.messageListCount>0 ? "available" : (AXIsProcessTrusted() && snapshot.scanned<=12 ? "ax_collapsed" : "unavailable")
            lastAXCounts=["nodes":snapshot.scanned,"bubbles":snapshot.bubbleCount,"virtual_cells":snapshot.virtualCount,
                          "titles":snapshot.titleCount,"message_lists":snapshot.messageListCount]
            if let click=click,click.sequence != clickSequence{return}
            if click==nil {
                if let title=snapshot.title {
                    let result=await observe(title,visible:[],pid:pid,source:"ax",force:false)
                    if let result=result{lastStatus=result["status"] as? String ?? "observed";lastSource="ax"}
                } else if Date().timeIntervalSince(lastHeaderOCR)>=2 {
                    // Current WeChat renderers may expose only window chrome.
                    // In that case poll the small heading, never the body. This
                    // keeps chat-switch/consent behavior available with existing
                    // screen permission and clearly reports OCR as the source.
                    lastHeaderOCR=Date()
                    let sequence=clickSequence
                    if Date().timeIntervalSince(aliasesLoaded)>120 {
                        let directory=await apiContacts()
                        if !directory.isEmpty{aliases=directory};aliasesLoaded=Date()
                    }
                    guard sequence==clickSequence,NSWorkspace.shared.frontmostApplication?.processIdentifier==pid else{return}
                    if let (_,_,_,_,title)=try? await capture(pid:pid,aliases:aliases,includeMessages:false) {
                        headerOCRCount+=1
                        guard sequence==clickSequence,NSWorkspace.shared.frontmostApplication?.processIdentifier==pid else{return}
                        let result=await observe(title,visible:[],pid:pid,source:"ocr",force:false)
                        if let result=result{lastStatus=result["status"] as? String ?? "observed";lastSource="ocr"}
                    }
                }
                return
            }
            guard let click=click else{return}
            do {
                var title=snapshot.title;var payload:[String:Any]?;var source="ax"
                var visibleTexts=snapshot.visibleTexts
                if click.messageRegion,let index=axMessageIndex(click.point,snapshot:snapshot){payload=axLocator(index:index,messages:snapshot.messages)}
                // Body OCR is used only when a deliberate click cannot be
                // resolved by the current AX tree. The timer sees heading only.
                if title==nil || (click.messageRegion && payload==nil) {
                    if Date().timeIntervalSince(aliasesLoaded)>120 {
                        let directory=await apiContacts()
                        guard click.sequence==clickSequence else{return}
                        if !directory.isEmpty{aliases=directory};aliasesLoaded=Date()
                    }
                    if let (window,_,lines,scale,ocrTitle)=try await capture(pid:pid,aliases:aliases,includeMessages:click.messageRegion,axTitle:title) {
                        title=ocrTitle;source="ocr"
                        visibleTexts=Array(lines.prefix(30)).map{String($0.text.prefix(800))}
                        if click.messageRegion,isMessageRegion(click.point,in:window.frame){payload=locator(click.point,window,lines,scale,target:ocrTitle)}
                    }
                }
                guard click.sequence==clickSequence,let currentTitle=title else{lastStatus="chat_not_visible";publish();return}
                let observation=await observe(currentTitle,visible:visibleTexts,pid:pid,source:source,force:true)
                guard click.sequence==clickSequence else{return}
                lastSource=source
                guard let observation=observation else{lastStatus="service_unavailable";publish();return}
                lastStatus=observation["status"] as? String ?? "observed"
                guard click.messageRegion,var selected=payload,
                      ["matched","recognized"].contains(observation["status"] as? String ?? ""),let target=observation["active_target"] as? String else{publish();return}
                // Re-read the title immediately before committing a location;
                // if the user switched the conversation while requests ran,
                // never attach old bubble text to the new contact.
                if source=="ax" {
                    let check=await Task.detached{axSnapshot(pid:pid,includeMessages:false)}.value
                    guard check.title==currentTitle,click.sequence==clickSequence else{pendingObserve=true;lastStatus="stale_click";publish();return}
                } else {
                    // A keyboard chat switch can occur without a mouse event.
                    // Confirm the fresh heading rather than trusting the first
                    // screenshot while the local contact lookup was in flight.
                    let checked=try await capture(pid:pid,aliases:aliases,includeMessages:false)
                    guard checked?.4==currentTitle,click.sequence==clickSequence else{pendingObserve=true;lastStatus="stale_click";publish();return}
                }
                selected["target"]=target;selected["contact_revision"]=observation["contact_revision"] ?? NSNull()
                selected["source"]=source;selected["location"]=["x":click.point.x,"y":click.point.y,"wechat_pid":Int(pid)]
                clicks+=1
                let result=await api("/api/history/select-position",selected,timeout:20)
                if click.sequence==clickSequence{lastStatus=result?["status"] as? String ?? "service_unavailable";publish()}
            }catch {lastStatus="capture_failed";publish()}
        }
    }
}

if CommandLine.arguments.contains("--ax-self-test") {
    let viewport=CGRect(x:300,y:140,width:600,height:500)
    let bubble=CGRect(x:370,y:220,width:250,height:60)
    let fullRow=CGRect(x:300,y:300,width:600,height:70)
    let dummy=AXUIElementCreateApplication(getpid())
    let messages=[AXMessage(element:dummy,text:"第一条合成内容",rect:bubble),AXMessage(element:dummy,text:"第二条合成内容",rect:fullRow)]
    let located=axLocator(index:1,messages:messages)
    let checks:[Bool]=[
        cleanChatTitle("  合成联系人  ")=="合成联系人",
        cleanChatTitle("")==nil,
        cleanChatTitle("标题\n消息")==nil,
        cleanChatTitle(String(repeating:"字",count:121))==nil,
        validAXMessage("好"),validAXMessage("123"),validAXMessage("🙂"),!validAXMessage("  "),
        axGeometricMessageIndex(point:CGPoint(x:420,y:240),viewport:viewport,rects:[bubble,fullRow])==0,
        axGeometricMessageIndex(point:CGPoint(x:420,y:330),viewport:viewport,rects:[bubble,fullRow])==1,
        axGeometricMessageIndex(point:CGPoint(x:320,y:330),viewport:viewport,rects:[bubble,fullRow])==nil,
        axGeometricMessageIndex(point:CGPoint(x:880,y:330),viewport:viewport,rects:[bubble,fullRow])==nil,
        axGeometricMessageIndex(point:CGPoint(x:420,y:190),viewport:viewport,rects:[bubble,fullRow])==nil,
        axGeometricMessageIndex(point:CGPoint(x:100,y:240),viewport:viewport,rects:[bubble,fullRow])==nil,
        axGeometricMessageIndex(point:CGPoint(x:420,y:240),viewport:viewport,rects:[bubble,bubble])==nil,
        (located?["clicked_texts"] as? [String])==["第二条合成内容"],
        (located?["before_texts"] as? [String])==["第一条合成内容"],
        (located?["after_texts"] as? [String])==[],
        axLocator(index:2,messages:messages)==nil
    ]
    print(String(data:json(["ax_self_test":checks.allSatisfy{$0},"checks":checks.count,"passed":checks.filter{$0}.count,"synthetic_only":true]),encoding:.utf8)!)
    exit(checks.allSatisfy{$0} ? 0 : 1)
}
if CommandLine.arguments.contains("--ax-probe") {
    // No activation, scrolling, capture, permission request, or real chat text.
    guard let wechat=NSWorkspace.shared.runningApplications.first(where:{wechatBundles.contains($0.bundleIdentifier ?? "")}) else {
        print(String(data:json(["ax_probe":true,"wechat_running":false,"accessibility":AXIsProcessTrusted(),"metadata_only":true]),encoding:.utf8)!);exit(0)
    }
    let axApp=AXUIElementCreateApplication(wechat.processIdentifier);AXUIElementSetMessagingTimeout(axApp,0.1)
    func probeSnapshot()->[String:Any] {
        let snapshot=axSnapshot(pid:wechat.processIdentifier)
        var childCounts:[String:Int]=[:]
        let window=axElement(axAttribute(axApp,kAXFocusedWindowAttribute)) ?? axElement(axAttribute(axApp,kAXMainWindowAttribute))
        for attribute in [kAXChildrenAttribute,"AXVisibleChildren","AXContents"] {
            childCounts["app_"+attribute]=(axAttribute(axApp,attribute) as? [AnyObject])?.count ?? 0
            if let window=window{childCounts["window_"+attribute]=(axAttribute(window,attribute) as? [AnyObject])?.count ?? 0}
        }
        return ["nodes":snapshot.scanned,"message_lists":snapshot.messageListCount,"title_elements":snapshot.titleCount,
                "bubbles":snapshot.bubbleCount,"virtual_cells":snapshot.virtualCount,"readable_messages":snapshot.messages.count,
                "current_chat_detected":snapshot.title != nil,"elapsed_ms":Int(snapshot.elapsed*1000),
                "roles":snapshot.roleCounts,"known_identifier_counts":snapshot.knownIdentifiers,"child_counts":childCounts]
    }
    var output:[String:Any]=["ax_probe":true,"wechat_running":true,"accessibility":AXIsProcessTrusted(),
                            "metadata_only":true,"screen_capture_used":false,"baseline":probeSnapshot()]
    if CommandLine.arguments.contains("--try-ax-enable"),AXIsProcessTrusted() {
        // AXManualAccessibility is documented by Electron. Enhanced UI is the
        // assistive-technology flag checked by Chromium and public Mac readers.
        // This explicit diagnostic restores prior values; normal observation
        // never changes app flags or implies that WeChat is an Electron app.
        var originals:[String:CFTypeRef]=[:];var results:[String:Any]=[:]
        for attribute in ["AXManualAccessibility","AXEnhancedUserInterface"] {
            let previous=axAttribute(axApp,attribute)
            let originalBoolean=(previous != nil && CFGetTypeID(previous!)==CFBooleanGetTypeID()) ? previous : nil
            let wasEnabled=(previous as? Bool)==true
            if wasEnabled{results[attribute]=["already_enabled":true,"set_attempted":false]}
            else {
                let result=AXUIElementSetAttributeValue(axApp,attribute as CFString,kCFBooleanTrue)
                results[attribute]=["already_enabled":false,"set_attempted":true,"set_error":result.rawValue]
                if result == .success{originals[attribute]=originalBoolean ?? kCFBooleanFalse}
            }
        }
        usleep(250_000)
        output["enablement_results"]=results;output["after_enablement"]=probeSnapshot()
        var restored:[String:Int]=[:]
        for (attribute,value) in originals{restored[attribute]=Int(AXUIElementSetAttributeValue(axApp,attribute as CFString,value).rawValue)}
        output["restore_errors"]=restored
    }
    print(String(data:json(output),encoding:.utf8)!);exit(0)
}

if CommandLine.arguments.contains("--stability-self-test") {
    let frame=CGRect(x:100,y:80,width:900,height:650)
    let line=Line(text:"合成测试消息",rect:CGRect(x:300,y:150,width:180,height:28))
    let checks:[Bool]=[
        isMessageRegion(CGPoint(x:500,y:300),in:frame),
        !isMessageRegion(CGPoint(x:200,y:300),in:frame),
        !isMessageRegion(CGPoint(x:500,y:100),in:frame),
        !isMessageRegion(CGPoint(x:500,y:640),in:frame),
        !isMessageRegion(CGPoint(x:1100,y:300),in:frame),
        messageLocator(local:CGPoint(x:360,y:160),lines:[line],scale:1) != nil,
        messageLocator(local:CGPoint(x:360,y:160),lines:[],scale:1) == nil,
        messageLocator(local:CGPoint(x:700,y:300),lines:[line],scale:1) == nil,
        messageLocator(local:CGPoint(x:512,y:160),lines:[line],scale:1) != nil,
        messageLocator(local:CGPoint(x:513,y:160),lines:[line],scale:1) == nil
    ]
    print(String(data:json(["stability_self_test":checks.allSatisfy{$0},"checks":checks.count,
                            "passed":checks.filter{$0}.count,"synthetic_only":true]),encoding:.utf8)!)
    exit(checks.allSatisfy{$0} ? 0 : 1)
}

if CommandLine.arguments.contains("--contact-self-test") {
    // Synthetic OCR metadata only: no app activation, screen capture, or API.
    func heading(_ text: String, x: CGFloat = 300) -> Line {
        Line(text: text, rect: CGRect(x:x,y:20,width:100,height:20))
    }
    let allowed = ["示例好友甲", "示例好友乙"]
    let checks: [Bool] = [
        recognizedContact([heading("示例好友甲")], imageWidth:1000, allowedTargets:allowed) == "示例好友甲",
        recognizedContact([heading("示例好友乙")], imageWidth:1000, allowedTargets:allowed) == "示例好友乙",
        recognizedContact([heading("其他好友")], imageWidth:1000, allowedTargets:allowed) == nil,
        recognizedContact([heading("示例好友乙的群聊")], imageWidth:1000, allowedTargets:allowed) == nil,
        recognizedContact([heading("示例好友甲"), heading("示例好友乙")], imageWidth:1000, allowedTargets:allowed) == nil,
        recognizedContact([heading("示例好友乙", x:50)], imageWidth:1000, allowedTargets:allowed) == nil,
        recognizedContact([heading("示例好友乙"), heading("示例好友乙")], imageWidth:1000, allowedTargets:allowed) == "示例好友乙",
        recognizedContact([heading("示例 好友乙")], imageWidth:1000, allowedTargets:allowed) == "示例好友乙",
        recognizedContact([heading("示例好友甲")], imageWidth:1000, allowedTargets:["示例好友乙"]) == nil,
        recognizedContact([heading("示例好友乙")], imageWidth:1000, allowedTargets:[]) == nil
    ]
    print(String(data:json(["contact_self_test":checks.allSatisfy{$0},"checks":checks.count,
                            "passed":checks.filter{$0}.count,"configured_contact_count":targets.count,
                            "synthetic_only":true]),encoding:.utf8)!)
    exit(checks.allSatisfy{$0} ? 0 : 1)
}

let app=NSApplication.shared;app.setActivationPolicy(.prohibited)
let observer=MainActor.assumeIsolated { Observer() }
if CommandLine.arguments.contains("--test-click") {
    Task { @MainActor in
        guard let wechat=NSWorkspace.shared.runningApplications.first(where:{$0.bundleIdentifier=="com.tencent.xinWeChat"}) else{exit(2)}
        wechat.activate(options: [])
        try? await Task.sleep(nanoseconds:700_000_000)
        let snapshot=await Task.detached{axSnapshot(pid:wechat.processIdentifier)}.value
        var chosen:CGPoint?
        if snapshot.title != nil,let message=snapshot.messages.first(where:{$0.text.count>=8}){chosen=CGPoint(x:message.rect.midX,y:message.rect.midY)}
        else {
            let aliases=await apiContacts()
            if let (window,_,lines,scale,_)=try? await capture(pid:wechat.processIdentifier,aliases:aliases.isEmpty ? targets : aliases),
               let line=lines.first(where:{normalize($0.text).count>=8}){chosen=CGPoint(x:window.frame.minX+line.rect.midX/scale,y:window.frame.minY+line.rect.midY/scale)}
        }
        guard let p=chosen else{print("{\"test_click_posted\":false,\"reason\":\"chat_or_text_not_visible\"}");exit(3)}
        wechat.activate(options: [])
        try? await Task.sleep(nanoseconds:150_000_000)
        CGEvent(mouseEventSource:nil,mouseType:.mouseMoved,mouseCursorPosition:p,mouseButton:.left)?.post(tap:.cghidEventTap)
        let all=CGWindowListCopyWindowInfo([.optionOnScreenOnly,.excludeDesktopElements],kCGNullWindowID) as? [[String:Any]] ?? []
        let top=all.first{item in guard let b=item[kCGWindowBounds as String] as? [String:Any],let r=CGRect(dictionaryRepresentation:b as CFDictionary) else{return false};return (item[kCGWindowOwnerPID as String] as? Int)==Int(wechat.processIdentifier) && r.contains(p)}
        print(String(data:json(["test_top_pid":top?[kCGWindowOwnerPID as String] ?? 0,"test_top_layer":top?[kCGWindowLayer as String] ?? -1,"wechat_pid":Int(wechat.processIdentifier)]),encoding:.utf8)!)
        CGEvent(mouseEventSource:nil,mouseType:.leftMouseDown,mouseCursorPosition:p,mouseButton:.left)?.post(tap:.cghidEventTap)
        CGEvent(mouseEventSource:nil,mouseType:.leftMouseUp,mouseCursorPosition:p,mouseButton:.left)?.post(tap:.cghidEventTap)
        print("{\"test_click_posted\":true,\"scope\":\"existing_target_message\"}");exit(0)
    }
} else {MainActor.assumeIsolated { observer.start() }}
app.run()
