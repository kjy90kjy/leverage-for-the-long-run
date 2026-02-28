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

  const { signal, signal_type, price, date, regime, prediction_text } = data;

  // 배경색
  if (signal === 1) {
    w.backgroundColor = new Color("#1abc9c"); // 초록색 (매수)
  } else {
    w.backgroundColor = new Color("#e74c3c"); // 빨강색 (관망)
  }

  w.setPadding(12, 12, 12, 12);

  // 헤더: 신호
  const signalText = w.addText(signal === 1 ? "🟢 BUY" : "🔴 HOLD");
  signalText.font = Font.boldSystemFont(32);
  signalText.textColor = new Color("white");
  signalText.lineLimit = 1;

  // 신호 타입
  const typeText = w.addText(signal_type || "관망");
  typeText.font = Font.systemFont(12);
  typeText.textColor = new Color("rgba(255,255,255,0.8)");
  typeText.lineLimit = 2;

  w.addSpacer(8);

  // 가격 정보
  const priceText = w.addText(`$${Math.round(price)}`);
  priceText.font = Font.boldSystemFont(20);
  priceText.textColor = new Color("white");

  const dateText = w.addText(date);
  dateText.font = Font.systemFont(11);
  dateText.textColor = new Color("rgba(255,255,255,0.7)");

  w.addSpacer(8);

  // 레짐
  const regimeText = w.addText(
    regime === "LOW" ? "❄️ Low Vol" : "🔥 High Vol"
  );
  regimeText.font = Font.systemFont(12);
  regimeText.textColor = new Color("rgba(255,255,255,0.9)");

  // 예측 정보 (있으면 표시)
  if (prediction_text) {
    w.addSpacer(6);
    const predText = w.addText("📈 " + prediction_text);
    predText.font = Font.systemFont(10);
    predText.textColor = new Color("rgba(255,255,255,0.8)");
    predText.lineLimit = 3;
  }

  // 시간 정보
  w.addSpacer(6);
  const timeText = w.addText(
    `Updated: ${new Date().toLocaleTimeString("ko-KR")}`
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
