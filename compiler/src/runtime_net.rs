//! Runtime support for the prelude types: JSON and the HTTP server.

pub const RUNTIME_NET: &str = r####"
#[allow(dead_code)]
mod ail_json {
    use super::U_Json as J;
    use super::ail_rt::AilMap;

    pub fn parse(s: &String) -> Result<J, String> {
        let mut p = P { b: s.as_bytes(), i: 0, depth: 0 };
        p.ws();
        let v = p.value()?;
        p.ws();
        if p.i != p.b.len() { return Err(p.err("trailing characters")); }
        Ok(v)
    }

    struct P<'a> { b: &'a [u8], i: usize, depth: u32 }

    impl<'a> P<'a> {
        fn err(&self, m: &str) -> String { format!("json: {m} at byte {}", self.i) }
        fn ws(&mut self) {
            while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\n' | b'\r' | b'\t') { self.i += 1; }
        }
        fn lit(&mut self, w: &str, v: J) -> Result<J, String> {
            if self.b[self.i..].starts_with(w.as_bytes()) { self.i += w.len(); Ok(v) } else { Err(self.err("invalid literal")) }
        }
        fn enter(&mut self) -> Result<(), String> {
            self.depth += 1;
            if self.depth > 256 { Err(self.err("nesting too deep")) } else { Ok(()) }
        }
        fn value(&mut self) -> Result<J, String> {
            if self.i >= self.b.len() { return Err(self.err("unexpected end")); }
            match self.b[self.i] {
                b'n' => self.lit("null", J::Null),
                b't' => self.lit("true", J::Bool(true)),
                b'f' => self.lit("false", J::Bool(false)),
                b'"' => Ok(J::Str(self.string()?)),
                b'[' => {
                    self.enter()?;
                    self.i += 1;
                    let mut v = Vec::new();
                    self.ws();
                    if self.b.get(self.i) == Some(&b']') { self.i += 1; self.depth -= 1; return Ok(J::Arr(v)); }
                    loop {
                        self.ws();
                        v.push(self.value()?);
                        self.ws();
                        match self.b.get(self.i) {
                            Some(b',') => self.i += 1,
                            Some(b']') => { self.i += 1; break; }
                            _ => return Err(self.err("expected ',' or ']'")),
                        }
                    }
                    self.depth -= 1;
                    Ok(J::Arr(v))
                }
                b'{' => {
                    self.enter()?;
                    self.i += 1;
                    let mut m = AilMap::default();
                    self.ws();
                    if self.b.get(self.i) == Some(&b'}') { self.i += 1; self.depth -= 1; return Ok(J::Obj(m)); }
                    loop {
                        self.ws();
                        if self.b.get(self.i) != Some(&b'"') { return Err(self.err("expected a string key")); }
                        let k = self.string()?;
                        self.ws();
                        if self.b.get(self.i) != Some(&b':') { return Err(self.err("expected ':'")); }
                        self.i += 1;
                        self.ws();
                        let v = self.value()?;
                        m.insert(k, v);
                        self.ws();
                        match self.b.get(self.i) {
                            Some(b',') => self.i += 1,
                            Some(b'}') => { self.i += 1; break; }
                            _ => return Err(self.err("expected ',' or '}'")),
                        }
                    }
                    self.depth -= 1;
                    Ok(J::Obj(m))
                }
                b'-' | b'0'..=b'9' => {
                    let st = self.i;
                    // Fast path: short plain integers are exact in f64 without the float parser.
                    let neg = self.b[st] == b'-';
                    let mut j = st + neg as usize;
                    let mut v: i64 = 0;
                    while j < self.b.len() && self.b[j].is_ascii_digit() && j - st < 16 {
                        v = v * 10 + (self.b[j] - b'0') as i64;
                        j += 1;
                    }
                    let digits = j - st - neg as usize;
                    let ends = j >= self.b.len() || !matches!(self.b[j], b'.' | b'e' | b'E' | b'0'..=b'9' | b'-' | b'+');
                    if digits > 0 && ends && !(digits > 1 && self.b[st + neg as usize] == b'0') {
                        self.i = j;
                        return Ok(J::Num(if neg { -(v as f64) } else { v as f64 }));
                    }
                    while self.i < self.b.len() && matches!(self.b[self.i], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9') { self.i += 1; }
                    std::str::from_utf8(&self.b[st..self.i]).ok().and_then(|t| t.parse::<f64>().ok()).map(J::Num)
                        .ok_or_else(|| self.err("invalid number"))
                }
                _ => Err(self.err("unexpected character")),
            }
        }
        fn hex4(&self, at: usize) -> Result<u32, String> {
            self.b.get(at..at + 4).and_then(|h| std::str::from_utf8(h).ok()).and_then(|h| u32::from_str_radix(h, 16).ok())
                .ok_or_else(|| self.err("bad unicode escape"))
        }
        fn string(&mut self) -> Result<String, String> {
            self.i += 1;
            let mut out = String::new();
            loop {
                let Some(&c) = self.b.get(self.i) else { return Err(self.err("unterminated string")) };
                match c {
                    b'"' => { self.i += 1; return Ok(out); }
                    b'\\' => {
                        let e = *self.b.get(self.i + 1).ok_or_else(|| self.err("bad escape"))?;
                        self.i += 2;
                        match e {
                            b'"' => out.push('"'),
                            b'\\' => out.push('\\'),
                            b'/' => out.push('/'),
                            b'b' => out.push('\u{8}'),
                            b'f' => out.push('\u{c}'),
                            b'n' => out.push('\n'),
                            b'r' => out.push('\r'),
                            b't' => out.push('\t'),
                            b'u' => {
                                let mut cp = self.hex4(self.i)?;
                                self.i += 4;
                                if (0xD800..0xDC00).contains(&cp) && self.b.get(self.i) == Some(&b'\\') && self.b.get(self.i + 1) == Some(&b'u') {
                                    let lo = self.hex4(self.i + 2)?;
                                    if (0xDC00..0xE000).contains(&lo) {
                                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                                        self.i += 6;
                                    }
                                }
                                out.push(char::from_u32(cp).unwrap_or('\u{FFFD}'));
                            }
                            _ => return Err(self.err("bad escape")),
                        }
                    }
                    c if c < 0x20 => return Err(self.err("control character in string")),
                    _ => {
                        let st = self.i;
                        while self.i < self.b.len() && self.b[self.i] != b'"' && self.b[self.i] != b'\\' && self.b[self.i] >= 0x20 { self.i += 1; }
                        out.push_str(std::str::from_utf8(&self.b[st..self.i]).map_err(|_| self.err("invalid UTF-8"))?);
                    }
                }
            }
        }
    }

    /// Direct encoding of typed values (structs implement it in the generated code).
    pub trait Enc { fn enc(&self, o: &mut String); }
    macro_rules! enc_int { ($($t:ty),*) => { $(impl Enc for $t { #[inline] fn enc(&self, o: &mut String) { use std::fmt::Write as _; let _ = write!(o, "{}", self); } })* } }
    enc_int!(i8, i16, i32, i64, u8, u16, u32, u64);
    impl Enc for f64 { fn enc(&self, o: &mut String) { num(*self, o) } }
    impl Enc for f32 { fn enc(&self, o: &mut String) { num(*self as f64, o) } }
    impl Enc for bool { fn enc(&self, o: &mut String) { o.push_str(if *self { "true" } else { "false" }) } }
    impl Enc for String { fn enc(&self, o: &mut String) { write_str(self, o) } }
    impl Enc for J { fn enc(&self, o: &mut String) { write(self, o) } }
    impl<T: Enc> Enc for Box<T> { fn enc(&self, o: &mut String) { (**self).enc(o) } }
    impl<T: Enc> Enc for Option<T> { fn enc(&self, o: &mut String) { match self { Some(v) => v.enc(o), None => o.push_str("null") } } }
    impl<T: Enc> Enc for Vec<T> { fn enc(&self, o: &mut String) { self.as_slice().enc(o) } }
    impl<T: Enc> Enc for [T] {
        fn enc(&self, o: &mut String) {
            o.push('[');
            for (i, x) in self.iter().enumerate() { if i > 0 { o.push(','); } x.enc(o); }
            o.push(']');
        }
    }
    impl<T: Enc> Enc for AilMap<String, T> {
        fn enc(&self, o: &mut String) {
            let mut kv: Vec<(&String, &T)> = self.iter().collect();
            kv.sort_unstable_by(|a, b| a.0.cmp(b.0));
            o.push('{');
            for (i, (k, v)) in kv.iter().enumerate() { if i > 0 { o.push(','); } write_str(k, o); o.push(':'); v.enc(o); }
            o.push('}');
        }
    }
    pub fn encode<T: Enc + ?Sized>(x: &T) -> String {
        let mut o = String::with_capacity(128);
        x.enc(&mut o);
        o
    }

    fn num(n: f64, o: &mut String) {
        use std::fmt::Write as _;
        if n.is_finite() && n.fract() == 0.0 && n.abs() < 1e15 {
            let _ = write!(o, "{}", n as i64);
        } else if n.is_finite() {
            let _ = write!(o, "{n}");
        } else {
            o.push_str("null")
        }
    }

    pub fn to_string(j: &J) -> String {
        let mut o = String::with_capacity(128);
        write(j, &mut o);
        o
    }

    fn write_str(s: &str, o: &mut String) {
        use std::fmt::Write as _;
        o.push('"');
        let b = s.as_bytes();
        let mut st = 0;
        for (i, &c) in b.iter().enumerate() {
            let esc = match c {
                b'"' => "\\\"",
                b'\\' => "\\\\",
                b'\n' => "\\n",
                b'\r' => "\\r",
                b'\t' => "\\t",
                0..=0x1f => "",
                _ => continue,
            };
            o.push_str(&s[st..i]);
            if esc.is_empty() {
                let _ = write!(o, "\\u{:04x}", c);
            } else {
                o.push_str(esc);
            }
            st = i + 1;
        }
        o.push_str(&s[st..]);
        o.push('"');
    }

    fn write(j: &J, o: &mut String) {
        match j {
            J::Null => o.push_str("null"),
            J::Bool(b) => o.push_str(if *b { "true" } else { "false" }),
            J::Num(n) => num(*n, o),
            J::Str(s) => write_str(s, o),
            J::Arr(v) => {
                o.push('[');
                for (i, x) in v.iter().enumerate() {
                    if i > 0 { o.push(','); }
                    write(x, o);
                }
                o.push(']');
            }
            J::Obj(m) => {
                o.push('{');
                // Keys are written sorted. Small objects sort on the stack, without allocating.
                if m.len() <= 16 {
                    let mut kv: [Option<(&String, &J)>; 16] = [None; 16];
                    let mut n = 0;
                    for (k, v) in m.iter() {
                        let mut p = n;
                        while p > 0 && kv[p - 1].unwrap().0 > k {
                            kv[p] = kv[p - 1];
                            p -= 1;
                        }
                        kv[p] = Some((k, v));
                        n += 1;
                    }
                    for (i, e) in kv[..n].iter().enumerate() {
                        let (k, v) = e.unwrap();
                        if i > 0 { o.push(','); }
                        write_str(k, o);
                        o.push(':');
                        write(v, o);
                    }
                } else {
                    let mut kv: Vec<(&String, &J)> = m.iter().collect();
                    kv.sort_unstable_by(|a, b| a.0.cmp(b.0));
                    for (i, (k, v)) in kv.iter().enumerate() {
                        if i > 0 { o.push(','); }
                        write_str(k, o);
                        o.push(':');
                        write(v, o);
                    }
                }
                o.push('}');
            }
        }
    }
}

#[allow(dead_code)]
mod ail_http {
    //! `http-next` hands the program one parsed request at a time, so handlers stay simple and
    //! single-threaded. Accepting connections and reading/parsing requests happens on a pool of
    //! I/O threads, so slow clients never stall the program and parsing runs in parallel.
    use super::U_Request as R;
    use super::ail_rt::AilMap;
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::sync::atomic::{AtomicI64, Ordering};
    use std::sync::mpsc::{sync_channel, Receiver};
    use std::sync::Arc;

    type Parsed = (String, String, AilMap<String, String>, AilMap<String, String>, String);

    struct Port { rx: Receiver<(TcpStream, Parsed)>, max_body: Arc<AtomicI64> }

    thread_local! {
        static PORTS: RefCell<HashMap<i64, Port>> = RefCell::new(HashMap::new());
        static PENDING: RefCell<HashMap<i64, TcpStream>> = RefCell::new(HashMap::new());
        static NEXT_ID: RefCell<i64> = RefCell::new(1);
    }
    const MAX_HEADER_BYTES: usize = 16 * 1024;
    const MAX_HEADERS: usize = 100;

    fn decode(s: &str) -> String {
        let b = s.as_bytes();
        let mut out = Vec::with_capacity(b.len());
        let mut i = 0;
        while i < b.len() {
            if b[i] == b'+' {
                out.push(b' ');
                i += 1;
            } else if b[i] == b'%' && i + 3 <= b.len() {
                match std::str::from_utf8(&b[i + 1..i + 3]).ok().and_then(|h| u8::from_str_radix(h, 16).ok()) {
                    Some(v) => { out.push(v); i += 3; }
                    None => { out.push(b'%'); i += 1; }
                }
            } else {
                out.push(b[i]);
                i += 1;
            }
        }
        String::from_utf8_lossy(&out).into_owned()
    }

    fn read_request(stream: &TcpStream, max_body: i64) -> Result<Parsed, String> {
        stream.set_read_timeout(Some(std::time::Duration::from_secs(10))).ok();
        let mut r = BufReader::new(stream);
        let mut line = String::new();
        let mut total = r.read_line(&mut line).map_err(|e| e.to_string())?;
        let mut parts = line.trim_end().splitn(3, ' ');
        let method = parts.next().unwrap_or("").to_string();
        let target = parts.next().unwrap_or("").to_string();
        if method.is_empty() || target.is_empty() { return Err("malformed request line".into()); }
        let mut headers: AilMap<String, String> = AilMap::default();
        loop {
            let mut h = String::new();
            let n = r.read_line(&mut h).map_err(|e| e.to_string())?;
            total += n;
            if total > MAX_HEADER_BYTES || headers.len() > MAX_HEADERS { return Err("headers too large".into()); }
            let h = h.trim_end();
            if n == 0 || h.is_empty() { break; }
            if let Some((k, v)) = h.split_once(':') {
                headers.insert(k.trim().to_ascii_lowercase(), v.trim().to_string());
            }
        }
        let len: usize = headers.get("content-length").and_then(|v| v.parse().ok()).unwrap_or(0);
        if len as i64 > max_body { return Err(format!("body of {len} bytes exceeds the limit of {max_body}")); }
        let mut body = vec![0u8; len];
        r.read_exact(&mut body).map_err(|e| e.to_string())?;
        let body = String::from_utf8(body).map_err(|_| "body is not valid UTF-8".to_string())?;
        let (path, qs) = match target.split_once('?') {
            Some((p, q)) => (p.to_string(), q.to_string()),
            None => (target.clone(), String::new()),
        };
        let mut query = AilMap::default();
        for kv in qs.split('&').filter(|s| !s.is_empty()) {
            let (k, v) = kv.split_once('=').unwrap_or((kv, ""));
            query.insert(decode(k), decode(v));
        }
        Ok((method, decode(&path), query, headers, body))
    }

    fn reply(mut s: &TcpStream, status: i64, content_type: &str, body: &str) -> std::io::Result<()> {
        let mut out = Vec::with_capacity(128 + body.len());
        let _ = write!(
            out,
            "HTTP/1.1 {status} {}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            reason(status),
            body.len()
        );
        out.extend_from_slice(body.as_bytes());
        s.write_all(&out)
    }

    fn listen(port: i64, max_body: i64) -> Result<Port, String> {
        let p = u16::try_from(port).map_err(|_| format!("invalid port {port}"))?;
        let ls = TcpListener::bind(("0.0.0.0", p)).map_err(|e| format!("cannot listen on port {port}: {e}"))?;
        let limit = Arc::new(AtomicI64::new(max_body));
        let (tx, rx) = sync_channel::<(TcpStream, Parsed)>(1024);
        let workers = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).clamp(2, 16);
        for _ in 0..workers {
            let ls = ls.try_clone().map_err(|e| e.to_string())?;
            let tx = tx.clone();
            let limit = limit.clone();
            std::thread::spawn(move || loop {
                let Ok((stream, _)) = ls.accept() else { continue };
                let _ = stream.set_nodelay(true);
                match read_request(&stream, limit.load(Ordering::Relaxed)) {
                    Ok(p) => {
                        if tx.send((stream, p)).is_err() { return; }
                    }
                    Err(e) => { let _ = reply(&stream, 400, "text/plain", &format!("bad request: {e}")); }
                }
            });
        }
        Ok(Port { rx, max_body: limit })
    }

    pub fn next(port: i64, max_body: i64) -> Result<R, String> {
        super::ail_rt::flush();
        let (stream, (method, path, query, headers, body)) = PORTS.with(|ports| -> Result<_, String> {
            let mut ports = ports.borrow_mut();
            if !ports.contains_key(&port) {
                ports.insert(port, listen(port, max_body)?);
            }
            let p = &ports[&port];
            p.max_body.store(max_body, Ordering::Relaxed);
            p.rx.recv().map_err(|_| format!("server on port {port} stopped"))
        })?;
        let id = NEXT_ID.with(|n| {
            let v = *n.borrow();
            *n.borrow_mut() = v + 1;
            v
        });
        PENDING.with(|p| p.borrow_mut().insert(id, stream));
        Ok(R { id, method, path, query, headers, body })
    }

    fn reason(code: i64) -> &'static str {
        match code {
            200 => "OK", 201 => "Created", 204 => "No Content", 301 => "Moved Permanently", 302 => "Found",
            304 => "Not Modified", 400 => "Bad Request", 401 => "Unauthorized", 403 => "Forbidden", 404 => "Not Found",
            405 => "Method Not Allowed", 409 => "Conflict", 413 => "Payload Too Large", 422 => "Unprocessable Entity",
            429 => "Too Many Requests", 500 => "Internal Server Error", 503 => "Service Unavailable", _ => "Status",
        }
    }

    pub fn respond(req: &R, status: i64, content_type: &String, body: &String) -> Result<(), String> {
        if !(100..=999).contains(&status) { return Err(format!("invalid status {status}")); }
        let stream = PENDING.with(|p| p.borrow_mut().remove(&req.id));
        let Some(s) = stream else { return Err(format!("request {} was already answered", req.id)) };
        reply(&s, status, content_type, body).map_err(|e| e.to_string())
    }
}
"####;
