const fs = require("node:fs");
const path = require("node:path");
const { pathToFileURL } = require("node:url");

const { chromium } = require("playwright");

function resolveChromiumExecutablePath() {
  const fromEnv =
    process.env.RESUME_PDF_EXECUTABLE_PATH || process.env.CHROME_PATH || "";
  if (fromEnv && fs.existsSync(fromEnv)) return fromEnv;

  if (process.platform === "darwin") {
    const candidates = [
      "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
      "/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary",
      "/Applications/Chromium.app/Contents/MacOS/Chromium",
    ];

    for (const candidate of candidates) {
      if (fs.existsSync(candidate)) return candidate;
    }
  }

  return undefined;
}

async function main() {
  const repoRoot = path.resolve(__dirname, "..");
  const inputPath = path.join(repoRoot, "resume.html");
  const outputPath = path.join(repoRoot, "assets", "resume.pdf");

  if (!fs.existsSync(inputPath)) {
    throw new Error(`Input not found: ${inputPath}`);
  }

  fs.mkdirSync(path.dirname(outputPath), { recursive: true });

  const launchArgs = [];
  if (process.env.CI) {
    launchArgs.push("--no-sandbox");
  }

  const executablePath = resolveChromiumExecutablePath();
  const launchOptions = launchArgs.length ? { args: launchArgs } : {};
  if (executablePath) launchOptions.executablePath = executablePath;

  const browser = await chromium.launch(
    Object.keys(launchOptions).length ? launchOptions : undefined,
  );

  try {
    const page = await browser.newPage();
    await page.goto(pathToFileURL(inputPath).href, { waitUntil: "networkidle" });
    await page.emulateMedia({ media: "print" });
    await page.evaluate(() => document.fonts?.ready);

    await page.pdf({
      path: outputPath,
      format: "A4",
      printBackground: true,
      preferCSSPageSize: true,
    });
  } finally {
    await browser.close();
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
