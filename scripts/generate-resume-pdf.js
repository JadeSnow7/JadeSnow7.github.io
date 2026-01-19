const fs = require("node:fs");
const path = require("node:path");
const { pathToFileURL } = require("node:url");

const { chromium } = require("playwright");

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

  const browser = await chromium.launch(
    launchArgs.length ? { args: launchArgs } : undefined,
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
