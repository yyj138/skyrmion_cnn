"""End-to-end browser check against the running local model service."""

from io import BytesIO
from pathlib import Path

from PIL import Image
from playwright.sync_api import expect, sync_playwright


ROOT = Path(__file__).resolve().parents[1]
REVIEW_DIR = ROOT / "platform" / "runtime" / "browser-review"
REVIEW_DIR.mkdir(parents=True, exist_ok=True)
URL = "http://127.0.0.1:8765/"


with sync_playwright() as playwright:
    browser = playwright.chromium.launch(channel="chrome", headless=True)

    desktop = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
    desktop.goto(URL)
    expect(desktop.locator("#service-status")).to_contain_text("模型已就绪")
    expect(desktop.locator("#result-state")).to_have_text("预测完成")
    expect(desktop.locator("#kind-value")).to_have_text("单 toron")
    expect(desktop.locator("#eta-value")).to_have_text("36")
    expect(desktop.locator("#reference-details")).to_be_visible()
    desktop.locator("#reference-details summary").click()
    expect(desktop.locator("#compare-voltage-reference")).to_have_text("1.500 V")
    desktop.screenshot(path=REVIEW_DIR / "desktop-comparison.png", full_page=True)
    desktop.locator("#reference-details summary").click()
    desktop.wait_for_timeout(250)
    desktop.screenshot(path=REVIEW_DIR / "desktop-auto.png", full_page=True)

    desktop.locator('[data-example-id="double"]').click()
    expect(desktop.locator("#kind-value")).to_have_text("双 toron")
    expect(desktop.locator("#result-state")).to_have_text("预测完成")
    desktop.locator("#reference-details summary").click()
    expect(desktop.locator("#compare-kind-reference")).to_have_text("双 toron")

    desktop.locator("#image-input").set_input_files(
        ROOT / "platform" / "assets" / "examples" / "single_eta36_U1p50_v2.png"
    )
    expect(desktop.locator("#kind-value")).to_have_text("单 toron")
    expect(desktop.locator("#result-state")).to_have_text("预测完成")
    expect(desktop.locator("#reference-details")).to_be_hidden()
    expect(desktop.locator("#scope-note-text")).to_contain_text("无法判断")

    blank = BytesIO()
    Image.new("L", (300, 300)).save(blank, format="PNG")
    desktop.locator("#image-input").set_input_files(
        {"name": "blank.png", "mimeType": "image/png", "buffer": blank.getvalue()}
    )
    expect(desktop.locator("#result-error")).to_contain_text("纹理不足")
    expect(desktop.locator("#kind-value")).to_have_text("—")

    wrong_size = BytesIO()
    Image.new("L", (200, 200)).save(wrong_size, format="PNG")
    desktop.locator("#image-input").set_input_files(
        {"name": "small.png", "mimeType": "image/png", "buffer": wrong_size.getvalue()}
    )
    expect(desktop.locator("#input-error")).to_contain_text("300×300")
    expect(desktop.locator("#result-state")).to_have_text("等待预测")

    mobile = browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=1)
    mobile.goto(URL)
    expect(mobile.locator("#kind-value")).to_have_text("单 toron")
    expect(mobile.locator("#result-state")).to_have_text("预测完成")
    expect(mobile.locator("#predict-button")).to_be_enabled()
    mobile.wait_for_timeout(250)
    mobile.screenshot(path=REVIEW_DIR / "mobile-auto.png", full_page=True)
    assert mobile.evaluate("document.documentElement.scrollWidth <= innerWidth")

    rapid = browser.new_page(viewport={"width": 1000, "height": 760})
    rapid.add_init_script("""
      const originalFetch = window.fetch.bind(window);
      let predictionCount = 0;
      window.fetch = (...args) => {
        if (String(args[0]).endsWith('/api/predict') && predictionCount++ === 0) {
          return new Promise((resolve, reject) => {
            setTimeout(() => originalFetch(...args).then(resolve, reject), 650);
          });
        }
        return originalFetch(...args);
      };
    """)
    rapid.goto(URL)
    expect(rapid.locator('[data-example-id="single"]')).to_have_attribute("aria-pressed", "true")
    rapid.locator('[data-example-id="double"]').click()
    expect(rapid.locator("#kind-value")).to_have_text("双 toron")
    rapid.wait_for_timeout(800)
    expect(rapid.locator("#kind-value")).to_have_text("双 toron")
    expect(rapid.locator("#result-state")).to_have_text("预测完成")

    browser.close()

print("Browser flow passed: auto prediction, reference labels, upload, rejection, mobile, and rapid switching")
