// 前端：選圖或開相機 -> 把圖片送給後端 /api/detect -> 把回傳的旋轉框畫出來

const $ = (id) => document.getElementById(id);           // 縮寫：用 id 找元素
const COLORS = { ship: "#1e88e5", vehicle: "#e53935", airplane: "#43a047", court: "#fb8c00", road: "#8e24aa" };

// 把圖片（Blob）POST 給後端，回傳 { width, height, detections }
async function detect(blob) {
  const form = new FormData();
  form.append("image", blob);
  form.append("conf", $("conf").value);
  const res = await fetch("/api/detect", { method: "POST", body: form });
  if (!res.ok) throw new Error((await res.json()).detail);  // 後端錯誤訊息放在 detail
  return res.json();
}

// 在畫布上畫旋轉框。畫布大小 = 圖片大小，所以座標可以直接使用
function drawBoxes(canvas, result) {
  const ctx = canvas.getContext("2d");
  const lw = Math.max(2, canvas.width / 300);  // 線寬隨圖片大小調整
  ctx.lineWidth = lw;
  ctx.font = `${lw * 6}px sans-serif`;
  for (const d of result.detections) {
    ctx.strokeStyle = ctx.fillStyle = COLORS[d.name];
    ctx.beginPath();
    d.points.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.closePath();
    ctx.stroke();
    ctx.fillText(`${d.name} ${Math.round(d.conf * 100)}%`, d.points[0][0], d.points[0][1]);
  }
  // 摘要，例如「vehicle ×30、ship ×2」
  const count = {};
  result.detections.forEach((d) => (count[d.name] = (count[d.name] || 0) + 1));
  $("msg").textContent = Object.entries(count).map(([k, n]) => `${k} ×${n}`).join("、") || "沒有偵測到物件";
}

// ---------- 方式一：上傳照片 ----------
async function runPhoto() {
  const file = $("file").files[0];
  if (!file) return;
  try {
    const img = new Image();
    img.src = URL.createObjectURL(file);
    await img.decode();                       // 等圖片載入完成
    const canvas = $("photo");
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    canvas.getContext("2d").drawImage(img, 0, 0);  // 先畫照片
    $("msg").textContent = "辨識中…";
    drawBoxes(canvas, await detect(file));         // 再畫偵測框
  } catch (e) {
    $("msg").textContent = "錯誤：" + e.message;
  }
}
$("file").onchange = runPhoto;
$("conf").oninput = () => ($("confText").textContent = $("conf").value);  // 顯示目前數值
$("conf").onchange = runPhoto;                                           // 放開滑桿後重新辨識

// ---------- 方式二：相機 ----------
let stream = null;  // 有值代表相機開著

async function runCamera() {
  const grab = document.createElement("canvas");  // 暫時用來截取目前畫面
  const overlay = $("overlay");
  while (stream) {                                // 一張辨識完才截下一張，不會塞爆後端
    grab.width = overlay.width = $("video").videoWidth;
    grab.height = overlay.height = $("video").videoHeight;
    grab.getContext("2d").drawImage($("video"), 0, 0);
    const blob = await new Promise((ok) => grab.toBlob(ok, "image/jpeg", 0.7));
    try {
      const result = await detect(blob);
      overlay.getContext("2d").clearRect(0, 0, overlay.width, overlay.height);  // 清掉上一輪的框
      drawBoxes(overlay, result);
    } catch (e) {
      $("msg").textContent = "錯誤：" + e.message;
    }
  }
}

$("camBtn").onclick = async () => {
  if (stream) {  // 已開啟 -> 關閉
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
    $("camBtn").textContent = "開啟相機";
    return;
  }
  try {
    // facingMode: "environment" = 手機後鏡頭；電腦會自動用內建鏡頭
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
  } catch (e) {
    $("msg").textContent = "無法開啟相機：" + e.message;
    return;
  }
  $("video").srcObject = stream;
  await $("video").play();
  $("camBtn").textContent = "關閉相機";
  runCamera();
};
