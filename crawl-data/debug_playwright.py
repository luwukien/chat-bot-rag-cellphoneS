from playwright.sync_api import sync_playwright
url='https://cellphones.com.vn/iphone-17-pro.html'
with sync_playwright() as p:
    browser=p.chromium.launch(headless=False)
    page=browser.new_page()
    page.goto(url, wait_until='domcontentloaded', timeout=60000)
    print('loaded domcontentloaded')
    button = page.locator('.button__show-modal-technical')
    print('button count', button.count())
    if button.is_visible():
        print('button visible')
        button.click()
        page.wait_for_selector('.teleport-modal', timeout=10000)
        print('modal count', page.locator('.teleport-modal').count())
        print('sections count', page.locator('section.technical-content-section').count())
        print('modal inner html sample', page.locator('.teleport-modal').inner_html()[:500])
    else:
        print('button not visible')
    browser.close()
