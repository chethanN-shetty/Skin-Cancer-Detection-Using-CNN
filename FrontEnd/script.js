const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");
const analyzeBtn = document.getElementById("analyzeBtn");
const resultBox = document.getElementById("resultBox");
const resultText = document.getElementById("resultText");
const progressContainer = document.querySelector(".progress-container");
const progressBar = document.querySelector(".progress-bar");

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      preview.src = e.target.result;
      preview.style.display = "block";
    };
    reader.readAsDataURL(file);
  }
});

analyzeBtn.addEventListener("click", () => {
  if (!fileInput.files[0]) {
    alert("Please upload an image first!");
    return;
  }

  resultBox.classList.add("hidden");
  progressContainer.classList.remove("hidden");
  progressBar.style.width = "0%";

  let progress = 0;
  const interval = setInterval(() => {
    progress += 10;
    progressBar.style.width = progress + "%";

    if (progress >= 100) {
      clearInterval(interval);
      progressContainer.classList.add("hidden");
      resultBox.classList.remove("hidden");

      const results = [
        "✅ Skin looks healthy! Stay sun-safe ☀️",
        "⚠️ Slight irregularity — keep checking 💬",
        "🚨 Potential risk detected — consult a dermatologist 💗",
      ];
      const randomResult = results[Math.floor(Math.random() * results.length)];
      resultText.innerText = randomResult;
    }
  }, 250);
});

// Fade-in animation
const sections = document.querySelectorAll('.fade-in');
function reveal() {
  const trigger = window.innerHeight * 0.85;
  sections.forEach(sec => {
    const rect = sec.getBoundingClientRect();
    if (rect.top < trigger) sec.classList.add('visible');
  });
}
window.addEventListener('scroll', reveal);
reveal();
