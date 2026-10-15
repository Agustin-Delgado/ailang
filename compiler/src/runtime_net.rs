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
                    _ => {
                        let st = self.i;
                        while self.i < self.b.len() && self.b[self.i] != b'"' && self.b[self.i] != b'\\' { self.i += 1; }
                        out.push_str(std::str::from_utf8(&self.b[st..self.i]).map_err(|_| self.err("invalid UTF-8"))?);
                    }
                }
            }
        }
    }

    pub fn to_string(j: &J) -> String {
        let mut o = String::new();
        write(j, &mut o);
        o
    }

    fn write_str(s: &str, o: &mut String) {
        o.push('"');
        for c in s.chars() {
            match c {
                '"' => o.push_str("\\\""),
                '\\' => o.push_str("\\\\"),
                '\n' => o.push_str("\\n"),
                '\r' => o.push_str("\\r"),
                '\t' => o.push_str("\\t"),
                c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
                c => o.push(c),
            }
        }
        o.push('"');
    }

    fn write(j: &J, o: &mut String) {
        match j {
            J::Null => o.push_str("null"),
            J::Bool(b) => o.push_str(if *b { "true" } else { "false" }),
            J::Num(n) => {
                if n.is_finite() && n.fract() == 0.0 && n.abs() < 1e15 {
                    o.push_str(&format!("{}", *n as i64))
                } else if n.is_finite() {
                    o.push_str(&format!("{n}"))
                } else {
                    o.push_str("null")
                }
            }
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
                let mut ks: Vec<&String> = m.keys().collect();
                ks.sort();
                o.push('{');
                for (i, k) in ks.iter().enumerate() {
                    if i > 0 { o.push(','); }
                    write_str(k, o);
                    o.push(':');
                    write(&m[*k], o);
                }
                o.push('}');
            }
        }
    }
}

#[allow(dead_code)]
mod ail_http {
    use super::U_Request as R;
    use super::ail_rt::AilMap;
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::{TcpListener, TcpStream};

    thread_local! {
        static LISTENERS: RefCell<HashMap<i64, TcpListener>> = RefCell::new(HashMap::new());
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
            } else if b[i] == b'%' && i + 2 < b.len() + 1 && i + 3 <= b.len() {
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

    type Parsed = (String, String, AilMap<String, String>, AilMap<String, String>, String);

    fn read_request(stream: &TcpStream, max_body: i64) -> Result<Parsed, String> {
        stream.set_read_timeout(Some(std::time::Duration::from_secs(10))).ok();
        let mut r = BufReader::new(stream.try_clone().map_err(|e| e.to_string())?);
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
        let (path, qs) = match target.split_once('?') {
            Some((p, q)) => (p.to_string(), q.to_string()),
            None => (target.clone(), String::new()),
        };
        let mut query = AilMap::default();
        for kv in qs.split('&').filter(|s| !s.is_empty()) {
            let (k, v) = kv.split_once('=').unwrap_or((kv, ""));
            query.insert(decode(k), decode(v));
        }
        Ok((method, decode(&path), query, headers, String::from_utf8_lossy(&body).into_owned()))
    }

    pub fn next(port: i64, max_body: i64) -> Result<R, String> {
        super::ail_rt::flush();
        let listener = LISTENERS.with(|l| -> Result<TcpListener, String> {
            let mut l = l.borrow_mut();
            if !l.contains_key(&port) {
                let p = u16::try_from(port).map_err(|_| format!("invalid port {port}"))?;
                let ls = TcpListener::bind(("0.0.0.0", p)).map_err(|e| format!("cannot listen on port {port}: {e}"))?;
                l.insert(port, ls);
            }
            l[&port].try_clone().map_err(|e| e.to_string())
        })?;
        loop {
            let (stream, _) = listener.accept().map_err(|e| e.to_string())?;
            match read_request(&stream, max_body) {
                Ok((method, path, query, headers, body)) => {
                    let id = NEXT_ID.with(|n| {
                        let v = *n.borrow();
                        *n.borrow_mut() = v + 1;
                        v
                    });
                    PENDING.with(|p| p.borrow_mut().insert(id, stream));
                    return Ok(R { id, method, path, query, headers, body });
                }
                Err(e) => {
                    let mut s = stream;
                    let msg = format!("bad request: {e}");
                    let _ = write!(s, "HTTP/1.1 400 Bad Request\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{msg}", msg.len());
                }
            }
        }
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
        let Some(mut s) = stream else { return Err(format!("request {} was already answered", req.id)) };
        let head = format!(
            "HTTP/1.1 {status} {}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            reason(status),
            body.len()
        );
        s.write_all(head.as_bytes()).and_then(|_| s.write_all(body.as_bytes())).and_then(|_| s.flush()).map_err(|e| e.to_string())
    }
}
"####;
