// 김째 S2 Dashboard — iOS Scriptable Medium Widget (Dark)
// Fetches Kim-jje S2 signal JSON from signal_server.py

const SERVER_URL = "http://100.66.137.23:8765/ndx_signal.json";

// ── Colors ──
const BG_COLOR = new Color("#0d0d0d");
const BG_SECONDARY = new Color("#1a1a1a");
const TEXT_PRIMARY = new Color("#e0e0e0");
const TEXT_SECONDARY = new Color("#888888");
const TEXT_DIM = new Color("#555555");

const COLOR_GREEN = new Color("#4caf50");
const COLOR_YELLOW = new Color("#ffc107");
const COLOR_RED = new Color("#f44336");
const COLOR_ORANGE = new Color("#ff9800");

function baseCodeColor(baseCode) {
  if (baseCode === 2) return COLOR_GREEN;
  if (baseCode === 1) return COLOR_YELLOW;
  return COLOR_RED;
}

function baseCodeEmoji(baseCode) {
  if (baseCode === 2) return "🟢";
  if (baseCode === 1) return "🟡";
  return "🔴";
}

function stageBar(stage, maxStage) {
  let bar = "";
  for (let i = 0; i < maxStage; i++) {
    bar += (i < stage) ? "■" : "□";
  }
  return bar;
}

function weightLabel(pct) {
  if (pct >= 100) return "풀진입";
  if (pct >= 90) return "90%";
  if (pct >= 80) return "80%";
  if (pct >= 10) return "10%";
  if (pct >= 5) return "5%";
  return "관망";
}

async function fetchData() {
  let req = new Request(SERVER_URL);
  req.timeoutInterval = 10;
  try {
    let json = await req.loadJSON();
    return json;
  } catch (e) {
    return null;
  }
}

function addText(stack, text, font, color, align) {
  let t = stack.addText(text);
  t.font = font;
  t.textColor = color;
  if (align === "right") t.rightAlignText();
  if (align === "center") t.centerAlignText();
  return t;
}

function addRow(stack) {
  let row = stack.addStack();
  row.layoutHorizontally();
  row.centerAlignContent();
  return row;
}

async function createWidget(data) {
  let w = new ListWidget();
  w.backgroundColor = BG_COLOR;
  w.setPadding(12, 14, 10, 14);

  if (!data) {
    addText(w, "김째 S2", Font.boldSystemFont(14), TEXT_PRIMARY);
    w.addSpacer(4);
    addText(w, "데이터 로드 실패", Font.systemFont(12), COLOR_RED);
    addText(w, "서버 확인 필요", Font.systemFont(10), TEXT_DIM);
    return w;
  }

  // ── Header: 김째 S2 + timestamp ──
  let header = addRow(w);
  let titleColor = baseCodeColor(data.base_code);
  addText(header, "김째 S2", Font.boldSystemFont(14), titleColor);
  header.addSpacer();

  // 날짜 + 시간 (짧게)
  let ts = data.timestamp || "";
  let timePart = ts.split(" ")[1] || "";
  let datePart = (data.date || "").slice(5);  // MM-DD
  addText(header, `${datePart} ${timePart.slice(0, 5)}`, Font.regularMonospacedSystemFont(10), TEXT_DIM);

  w.addSpacer(6);

  // ── QQQ line ──
  let qqqRow = addRow(w);
  addText(qqqRow, "QQQ", Font.boldMonospacedSystemFont(10), TEXT_SECONDARY);
  qqqRow.addSpacer(4);
  addText(qqqRow, `$${data.qqq_price}`, Font.regularMonospacedSystemFont(11), TEXT_PRIMARY);
  qqqRow.addSpacer(6);
  let gcSymbol = data.golden_cross ? ">" : "<";
  let gcColor = data.golden_cross ? COLOR_GREEN : COLOR_RED;
  let ma3Str = data.qqq_ma3 != null ? Math.round(data.qqq_ma3) : "—";
  let ma161Str = data.qqq_ma161 != null ? Math.round(data.qqq_ma161) : "—";
  addText(qqqRow, `MA3 ${ma3Str} ${gcSymbol} MA161 ${ma161Str}`, Font.regularMonospacedSystemFont(9), gcColor);

  w.addSpacer(2);

  // ── TQQQ line ──
  let tqqqRow = addRow(w);
  addText(tqqqRow, "TQQQ", Font.boldMonospacedSystemFont(10), TEXT_SECONDARY);
  tqqqRow.addSpacer(4);
  addText(tqqqRow, `$${data.tqqq_price}`, Font.regularMonospacedSystemFont(11), TEXT_PRIMARY);
  tqqqRow.addSpacer(6);
  let ma200Str = data.tqqq_ma200 != null ? data.tqqq_ma200.toFixed(1) : "—";
  addText(tqqqRow, `MA200 $${ma200Str}`, Font.regularMonospacedSystemFont(9), TEXT_DIM);

  w.addSpacer(6);

  // ── 이격도 + baseCode ──
  let distRow = addRow(w);
  let dist200Str = data.dist200 != null ? `${data.dist200}%` : "—";
  let distColor = (data.dist200 != null && data.dist200 >= 101) ? COLOR_GREEN : COLOR_RED;
  addText(distRow, `이격도 ${dist200Str}`, Font.boldMonospacedSystemFont(11), distColor);
  distRow.addSpacer(8);
  let bcColor = baseCodeColor(data.base_code);
  addText(distRow, `base: ${data.base_code}`, Font.regularMonospacedSystemFont(10), bcColor);

  w.addSpacer(3);

  // ── 과열 프로그레스 + 비중 ──
  let ohRow = addRow(w);
  let barStr = stageBar(data.overheat_stage, 4);
  addText(ohRow, barStr, Font.boldMonospacedSystemFont(13), COLOR_ORANGE);
  ohRow.addSpacer(6);
  addText(ohRow, `Stage ${data.overheat_stage}`, Font.regularMonospacedSystemFont(10), TEXT_SECONDARY);
  ohRow.addSpacer(8);

  let wPct = data.final_weight_pct;
  let wColor = wPct >= 80 ? COLOR_GREEN : (wPct >= 10 ? COLOR_YELLOW : COLOR_RED);
  addText(ohRow, `비중 ${wPct}%`, Font.boldMonospacedSystemFont(11), wColor);

  w.addSpacer(4);

  // ── 경고 뱃지 (vol_locked, stop_hit, spy_bear) ──
  let alerts = [];
  if (data.vol_locked) alerts.push("VOL🔒");
  if (data.stop_hit) alerts.push("STOP🛑");
  if (!data.spy_bull) alerts.push("SPY🐻");
  if (data.reentry_lock) alerts.push("RE🔒");

  if (alerts.length > 0) {
    let alertRow = addRow(w);
    addText(alertRow, alerts.join("  "), Font.boldMonospacedSystemFont(9), COLOR_RED);
    w.addSpacer(2);
  }

  // ── 구분선 ──
  let sepRow = addRow(w);
  addText(sepRow, "─".repeat(30), Font.regularMonospacedSystemFont(6), TEXT_DIM);

  w.addSpacer(2);

  // ── 다음 스테이지 전환 ──
  if (data.next_overheat_up) {
    let upRow = addRow(w);
    let nxt = data.next_overheat_up;
    addText(upRow, `↑ S${nxt.stage}: $${nxt.tqqq_price} (${nxt.dist200}%)`, Font.regularMonospacedSystemFont(9), COLOR_ORANGE);
  }
  if (data.next_overheat_down) {
    let dnRow = addRow(w);
    let nxt = data.next_overheat_down;
    addText(dnRow, `↓ S${nxt.stage}: $${nxt.tqqq_price} (${nxt.dist200}%)`, Font.regularMonospacedSystemFont(9), new Color("#64b5f6"));
  }
  if (data.qqq_cross_price) {
    let gcRow = addRow(w);
    addText(gcRow, `GC교차: QQQ $${data.qqq_cross_price}`, Font.regularMonospacedSystemFont(9), TEXT_DIM);
  }

  return w;
}

// ── Main ──
let data = await fetchData();
let widget = await createWidget(data);

if (config.runsInWidget) {
  Script.setWidget(widget);
} else {
  await widget.presentMedium();
}

Script.complete();
