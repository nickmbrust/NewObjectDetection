
import time


from selenium import webdriver
from selenium.webdriver import ActionChains
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait


service = ChromeService(executable_path=ChromeDriverManager().install())
options = ChromeOptions()
# When no data directory and profile are specified Chrome creates a new temporary profile in /tmp
# NOTE: close all chrome windows prior to running selenium on a specified profile
# options.add_argument("user-data-dir=/home/w029ecp/.config/google-chrome")
# options.add_argument("--profile-directory=Profile 4")

headless = False
if headless:
    options.add_argument("--headless")
# all settings must be set by this point or another driver needs to be made
driver = webdriver.Chrome(service=service, options=options)

searchStr = "epiphone guitar"

driver.get("https://images.google.com/")

# get search bar and enter search
searchbar = driver.find_element(By.CSS_SELECTOR, "[title='Search']")
searchbar.send_keys(searchStr + "\n")

# save url to return to search later
searchURL = driver.current_url

# returns list of image elements, see how "CSS selectors" work for more details
def getImages():
    return driver.find_elements(By.CSS_SELECTOR, ".islrc .isv-r img")

images = getImages()

for x in range(100):
    with open(f"datasets/{searchStr}{x}.png", 'wb') as f:
        # get link
        src = images[x].get_attribute('src')
        # open just that image
        driver.get(src)
        # take a screenshot and save
        f.write(driver.find_element(By.CSS_SELECTOR, "img").screenshot_as_png)

    # return to previous page
    driver.get(searchURL)

    # re-find all images because all of them are stale because we loaded a new page
    images = getImages()

    # footer = driver.find_element(By.TAG_NAME, "footer")
    # delta_y = images[-1].rect['y']
    ActionChains(driver)\
        .scroll_by_amount(0, 1000000)\
        .perform()

    time.sleep(2)

    


print("done, sleeping...")
while True:
    time.sleep(1)
