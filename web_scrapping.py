from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys

# website='https://www.windguru.cz/734'
# driver=webdriver.Chrome()
# driver.get(website)
# wait = WebDriverWait(driver, 30)
# xpath='//*[@id="forecasts-page-content"]/div[3]/div/div[3]/div[2]/div[3]'
# a=wait.until(EC.element_to_be_clickable((By.XPATH, xpath)))
# a=driver.find_elements(By.XPATH,xpath)


# website='https://www.windguru.cz/734'
# driver=webdriver.Chrome()
# driver.get(website)
# wait = WebDriverWait(driver, 30)
# xpath='//*[@id="searchspot"]'
# # b=wait.until(EC.element_to_be_clickable((By.XPATH, xpath)))
# b=wait.until(EC.visibility_of_all_elements_located((By.XPATH, xpath)))

# b[0].click()
# b[0].send_keys('haifa')
# time.sleep(1)
# b[0].send_keys(Keys.ENTER)
# time.sleep(5)
# driver.quit()


website='https://moodle2324.technion.ac.il'
driver=webdriver.Chrome()
driver.get(website)
wait = WebDriverWait(driver, 30)
xpath='//*[@id="usernavigation"]/div[3]/div/span/a'
# b=wait.until(EC.element_to_be_clickable((By.XPATH, xpath)))
b=wait.until(EC.visibility_of_all_elements_located((By.XPATH, xpath)))
b[0].click()
xpath='//*[@id="page-login-index"]'
# b=wait.until(EC.element_to_be_clickable((By.XPATH, xpath)))
b=wait.until(EC.visibility_of_all_elements_located((By.XPATH, xpath)))
b[0].click()



b=wait.until(EC.element_to_be_clickable((By.CLASS_NAME, xpath)))
print(b)
# b[0].send_keys('haifa')
# time.sleep(1)
# b[0].send_keys(Keys.ENTER)
time.sleep(5)
driver.quit()
    # for each 
   