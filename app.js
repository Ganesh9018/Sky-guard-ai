/* ============================================================
   SkyWatch — Shared App Script
   Custom cursor, background color shift, scroll reveals,
   stat counters, auth (signup/login) flow with localStorage
   ============================================================ */

/* ---------------- Custom cursor ---------------- */
(function initCursor() {
  const dot = document.getElementById("cursorDot");
  const ring = document.getElementById("cursorRing");
  if (!dot || !ring) return;

  let mx = window.innerWidth / 2;
  let my = window.innerHeight / 2;
  let rx = mx;
  let ry = my;

  document.addEventListener("mousemove", (e) => {
    mx = e.clientX;
    my = e.clientY;
    dot.style.left = mx + "px";
    dot.style.top = my + "px";

    // Background color shift follows the mouse
    const sx = (mx / window.innerWidth) * 100;
    const sy = (my / window.innerHeight) * 100;
    document.body.style.setProperty("--shift-x", sx + "%");
    document.body.style.setProperty("--shift-y", sy + "%");
  });

  (function loop() {
    // Ring trails behind the dot for a smooth effect
    rx += (mx - rx) * 0.16;
    ry += (my - ry) * 0.16;
    ring.style.left = rx + "px";
    ring.style.top = ry + "px";
    requestAnimationFrame(loop);
  })();

  // Grow ring over interactive elements
  const hoverables = "a, button, input, label, .hover-lift, .feature-card, .step, select, textarea";
  document.addEventListener("mouseover", (e) => {
    if (e.target.closest(hoverables)) ring.classList.add("hovering");
  });
  document.addEventListener("mouseout", (e) => {
    if (e.target.closest(hoverables)) ring.classList.remove("hovering");
  });
  document.addEventListener("mousedown", () => ring.classList.add("pressing"));
  document.addEventListener("mouseup", () => ring.classList.remove("pressing"));
})();

/* ---------------- Spotlight follows cursor inside cards ---------------- */
document.querySelectorAll(".spotlight").forEach((card) => {
  card.addEventListener("mousemove", (e) => {
    const r = card.getBoundingClientRect();
    card.style.setProperty("--mx", (e.clientX - r.left) + "px");
    card.style.setProperty("--my", (e.clientY - r.top) + "px");
  });
});

/* ---------------- Scroll reveal ---------------- */
(function initReveal() {
  const els = document.querySelectorAll(".reveal");
  if (!els.length) return;
  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((en) => {
        if (en.isIntersecting) {
          en.target.classList.add("visible");
          io.unobserve(en.target);
        }
      });
    },
    { threshold: 0.12 }
  );
  els.forEach((el) => io.observe(el));
})();

/* ---------------- Animated stat counters ---------------- */
(function initCounters() {
  const counters = document.querySelectorAll("[data-count]");
  if (!counters.length) return;
  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((en) => {
        if (!en.isIntersecting) return;
        const el = en.target;
        const target = parseInt(el.dataset.count, 10) || 0;
        const dur = 1400;
        const t0 = performance.now();
        (function tick(t) {
          const p = Math.min((t - t0) / dur, 1);
          el.textContent = Math.round(target * (1 - Math.pow(1 - p, 3)));
          if (p < 1) requestAnimationFrame(tick);
        })(t0);
        io.unobserve(el);
      });
    },
    { threshold: 0.4 }
  );
  counters.forEach((c) => io.observe(c));
})();

/* ============================================================
   AUTH — signup / login with localStorage
   ============================================================ */
const SkyAuth = {
  usersKey: "skywatch_users",
  sessionKey: "skywatch_session",

  users() {
    try { return JSON.parse(localStorage.getItem(this.usersKey)) || []; }
    catch { return []; }
  },
  saveUsers(u) { localStorage.setItem(this.usersKey, JSON.stringify(u)); },

  signup(name, email, password) {
    const users = this.users();
    email = email.trim().toLowerCase();
    if (users.some((u) => u.email === email)) {
      return { ok: false, msg: "An account with this email already exists." };
    }
    users.push({ name: name.trim(), email, password, created: Date.now() });
    this.saveUsers(users);
    return { ok: true, msg: "Account created! You can now sign in." };
  },

  login(email, password) {
    email = email.trim().toLowerCase();
    const user = this.users().find(
      (u) => u.email === email && u.password === password
    );
    if (!user) return { ok: false, msg: "Invalid email or password." };
    localStorage.setItem(this.sessionKey, JSON.stringify({
      name: user.name, email: user.email, at: Date.now()
    }));
    return { ok: true, msg: "Welcome back, " + user.name + "!" };
  },

  session() {
    try { return JSON.parse(localStorage.getItem(this.sessionKey)); }
    catch { return null; }
  },
  logout() { localStorage.removeItem(this.sessionKey); },

  /* If already signed in, skip straight to dashboard */
  redirectIfSignedIn() {
    if (this.session()) window.location.href = "dashboard.html";
  }
};

window.SkyAuth = SkyAuth;

/* ---------------- Login page wiring ---------------- */
(function initAuthPage() {
  const tabs = document.getElementById("authTabs");
  if (!tabs) return; // not on the login page

  SkyAuth.redirectIfSignedIn();

  const tabLogin = document.getElementById("tabLogin");
  const tabSignup = document.getElementById("tabSignup");
  const loginForm = document.getElementById("loginForm");
  const signupForm = document.getElementById("signupForm");
  const loginMsg = document.getElementById("loginMsg");
  const signupMsg = document.getElementById("signupMsg");
  const strengthBar = document.getElementById("strengthBar");

  function showMode(mode) {
    if (mode === "signup") {
      tabs.classList.add("signup");
      tabLogin.classList.remove("active");
      tabSignup.classList.add("active");
      loginForm.classList.remove("active");
      signupForm.classList.add("active");
    } else {
      tabs.classList.remove("signup");
      tabLogin.classList.add("active");
      tabSignup.classList.remove("active");
      signupForm.classList.remove("active");
      loginForm.classList.add("active");
    }
  }

  // Support ?mode=signup deep link
  const params = new URLSearchParams(location.search);
  showMode(params.get("mode") === "signup" ? "signup" : "login");

  tabLogin.addEventListener("click", () => showMode("login"));
  tabSignup.addEventListener("click", () => showMode("signup"));
  document.getElementById("gotoSignup").addEventListener("click", (e) => {
    e.preventDefault(); showMode("signup");
  });
  document.getElementById("gotoLogin").addEventListener("click", (e) => {
    e.preventDefault(); showMode("login");
  });

  function flash(el, type, text) {
    el.className = "form-msg " + type;
    el.textContent = text;
    if (type === "error") {
      setTimeout(() => { el.className = "form-msg"; }, 3500);
    }
  }

  // Password strength meter
  document.getElementById("signupPassword").addEventListener("input", (e) => {
    const v = e.target.value;
    let score = 0;
    if (v.length >= 6) score++;
    if (v.length >= 10) score++;
    if (/[A-Z]/.test(v) && /[a-z]/.test(v)) score++;
    if (/\d/.test(v)) score++;
    if (/[^A-Za-z0-9]/.test(v)) score++;
    const levels = [
      { w: "8%",  c: "#ff4d5e" },
      { w: "25%", c: "#ff4d5e" },
      { w: "45%", c: "#ffb020" },
      { w: "65%", c: "#ffb020" },
      { w: "85%", c: "#2fd573" },
      { w: "100%", c: "#2fd573" }
    ];
    strengthBar.style.width = levels[score].w;
    strengthBar.style.background = levels[score].c;
  });

  /* ---- Signup submit ---- */
  signupForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const name = document.getElementById("signupName").value;
    const email = document.getElementById("signupEmail").value;
    const pass = document.getElementById("signupPassword").value;
    const confirm = document.getElementById("signupConfirm").value;
    const agree = document.getElementById("agreeTerms").checked;

    if (!name || !email || !pass) return flash(signupMsg, "error", "Please fill in all fields.");
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) return flash(signupMsg, "error", "Please enter a valid email address.");
    if (pass.length < 6) return flash(signupMsg, "error", "Password must be at least 6 characters.");
    if (pass !== confirm) return flash(signupMsg, "error", "Passwords do not match.");
    if (!agree) return flash(signupMsg, "error", "Please agree to the Terms to continue.");

    const res = SkyAuth.signup(name, email, pass);
    if (!res.ok) return flash(signupMsg, "error", res.msg);

    flash(signupMsg, "success", res.msg + " Redirecting to sign in…");
    signupForm.reset();
    strengthBar.style.width = "0";
    setTimeout(() => {
      showMode("login");
      document.getElementById("loginEmail").value = email;
      flash(loginMsg, "success", "Account created — enter your password to sign in.");
    }, 1200);
  });

  /* ---- Login submit ---- */
  loginForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const email = document.getElementById("loginEmail").value;
    const pass = document.getElementById("loginPassword").value;

    if (!email || !pass) return flash(loginMsg, "error", "Please enter your email and password.");

    const res = SkyAuth.login(email, pass);
    if (!res.ok) return flash(loginMsg, "error", res.msg);

    flash(loginMsg, "success", res.msg + " Launching dashboard…");
    setTimeout(() => { window.location.href = "dashboard.html"; }, 900);
  });
})();