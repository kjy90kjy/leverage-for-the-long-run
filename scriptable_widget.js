// Scriptable Widget: NDX Daily Signal
// iOS 홈화면 위젯으로 신호 실시간 표시
//
// 설치 방법:
// 1. iOS에서 Scriptable 앱 설치
// 2. 아래 코드 복사 → Scriptable에서 "+" → 코드 붙여넣기
// 3. 홈화면에 위젯 추가: 길게 누르기 → 위젯 추가 → Scriptable → 스크립트 선택

// ═══════════════════════════════════════════════════════════════
// 설정
// ═══════════════════════════════════════════════════════════════

// 1. 파이썬 스크립트 결과를 저장할 URL
// (아래에서 설명: Google Sheets 또는 자신의 웹서버)
const DATA_URL = "https://YOUR_SERVER.com/ndx_signal.json";

// 또는 로컬 파일 경로 (Mac에서만 가능)
// const DATA_PATH = "/Users/user/path/to/output/ndx_signal.json";

// ═══════════════════════════════════════════════════════════════
// 위젯 메인 로직
// ═══════════════════════════════════════════════════════════════

async function fetchSignalData() {
  try {
    let data;

    // 방법 1: URL에서 JSON 다운로드 (권장)
    if (DATA_URL && DATA_URL !== "https://YOUR_SERVER.com/ndx_signal.json") {
      const req = new Request(DATA_URL);
      req.timeoutInterval = 10;
      const response = await req.loadJSON();
      data = response;
    } else {
      // 방법 2: 로컬 파일 (Mac에서만)
      const fm = FileManager.local();
      const contents = fm.readString(DATA_PATH);
      data = JSON.parse(contents);
    }

    return data;
  } catch (e) {
    console.log("Error fetching data: " + e);
    return null;
  }
}

function createWidget(data) {
  const w = new ListWidget();

  if (!data) {
    w.backgroundColor = new Color("#2c3e50");
    w.addText("⚠️ No Data");
    return w;
  }

  const {
    signal,
    signal_type,
    price,
    date,
    regime,
    current_fast_ma,
    current_slow_ma,
    next_signal_price,
    price_change_needed,
    price_pct_change,
    crossover_direction,
  } = data;

  // 배경색
  if (signal === 1) {
    w.backgroundColor = new Color("#1abc9c"); // 초록색 (매수)
  } else {
    w.backgroundColor = new Color("#e74c3c"); // 빨강색 (관망)
  }

  w.setPadding(10, 10, 10, 10);

  // ═══ 현재 신호 ═══
  const signalText = w.addText(signal === 1 ? "🟢 BUY" : "🔴 HOLD");
  signalText.font = Font.boldSystemFont(28);
  signalText.textColor = new Color("white");
  signalText.lineLimit = 1;

  // 신호 타입 + 레짐
  const regimeText = w.addText(
    (regime === "LOW" ? "❄️ " : "🔥 ") + (signal_type || "관망")
  );
  regimeText.font = Font.systemFont(11);
  regimeText.textColor = new Color("rgba(255,255,255,0.9)");
  regimeText.lineLimit = 1;

  w.addSpacer(6);

  // ═══ 가격 정보 ═══
  const priceText = w.addText(`$${Math.round(price)}`);
  priceText.font = Font.boldSystemFont(18);
  priceText.textColor = new Color("white");

  const dateText = w.addText(date);
  dateText.font = Font.systemFont(10);
  dateText.textColor = new Color("rgba(255,255,255,0.7)");

  w.addSpacer(6);

  // ═══ 현재 MA ═══
  const maContainer = w.addStack();
  maContainer.layoutHorizontally();

  const fastLabel = maContainer.addText(
    `F: $${Math.round(current_fast_ma)}`
  );
  fastLabel.font = Font.systemFont(9);
  fastLabel.textColor = new Color("rgba(255,255,255,0.8)");

  maContainer.addSpacer();

  const slowLabel = maContainer.addText(
    `S: $${Math.round(current_slow_ma)}`
  );
  slowLabel.font = Font.systemFont(9);
  slowLabel.textColor = new Color("rgba(255,255,255,0.8)");

  w.addSpacer(8);

  // ═══ 다음 신호 조건 ═══
  const nextSigText = w.addText("📍 NEXT SIGNAL:");
  nextSigText.font = Font.boldSystemFont(10);
  nextSigText.textColor = new Color("rgba(255,255,255,0.95)");

  const triggerText = w.addText(
    `$${Math.round(next_signal_price)} ${
      price_change_needed > 0
        ? `(+${price_pct_change.toFixed(2)}%)`
        : `(${price_pct_change.toFixed(2)}%)`
    }`
  );
  triggerText.font = Font.boldSystemFont(12);
  triggerText.textColor = new Color("rgba(255,255,255,0.95)");

  const crossoverSmall = w.addText(crossover_direction);
  crossoverSmall.font = Font.systemFont(8);
  crossoverSmall.textColor = new Color("rgba(255,255,255,0.7)");
  crossoverSmall.lineLimit = 1;

  w.addSpacer(4);

  // ═══ 시간 정보 ═══
  const timeText = w.addText(
    `${new Date().toLocaleTimeString("ko-KR", {
      hour: "2-digit",
      minute: "2-digit",
    })}`
  );
  timeText.font = Font.systemFont(8);
  timeText.textColor = new Color("rgba(255,255,255,0.6)");

  return w;
}

// ═══════════════════════════════════════════════════════════════
// 위젯 실행
// ═══════════════════════════════════════════════════════════════

const data = await fetchSignalData();
const widget = createWidget(data);

// 위젯 크기 설정
const size = config.widgetFamily || "medium";
widget.presentSmall(); // small / medium / large

Script.setWidget(widget);
Script.complete();
