from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time

# WebDriver 설정
options = webdriver.ChromeOptions()
options.add_argument('--headless')  # 브라우저를 표시하지 않으려면 주석 해제
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

# Chrome WebDriver 초기화
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

try:
    # URL 접속
    driver.get("https://aimee.kr/rank")

    # 성분별 버튼 클릭
    nutrient_button = WebDriverWait(driver, 20).until(
        # EC.element_to_be_clickable((By.CSS_SELECTOR, 'button[data-code="RANK_CATEGORY_NUTRIENT"]'))
        EC.element_to_be_clickable((By.CSS_SELECTOR, 'button[data-code="RANK_CATEGORY_PURPOSE"]'))
    )
    nutrient_button.click()

    # 드랍다운 버튼 클릭하여 영양소 목록 표시
    dropdown_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR, 'button.DropdownButton__StyledDropdownButton-sc-yyyg8f-0'))
    )
    dropdown_button.click()

    # 모든 영양소 체크박스 요소 가져오기
    nutrient_items = driver.find_elements(By.CSS_SELECTOR, 'ul.rank-modal__ContentList-sc-v5tdlw-1 li input[type="checkbox"]')
    nutrient_names = [item.get_attribute('data-purpose-code') for item in nutrient_items]
    print(nutrient_names)
    
    for nutrient in nutrient_names:
        print(f'\nProcessing nutrient: {nutrient}')

        # 현재 선택된 영양소의 체크박스 선택
        current_nutrient_checkbox = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, f'input[data-purpose-code="{nutrient}"]'))
        )
        current_nutrient_checkbox.click()

        # 제품 리스트가 로드될 때까지 대기
        time.sleep(3)  # 필요에 따라 적절한 시간으로 조정

        # 상위 10개 제품을 처리
        for rank in range(1, 11):
            product_items = driver.find_elements(By.CSS_SELECTOR, 'ul.RankedProducts__RankedProductList-sc-3kqhpt-4 li')
            # print(product_items)
            # time.sleep(10)
            # print(product_items)
            if rank <= len(product_items):
                product_item = product_items[rank - 1]  # 순서에 맞는 li 요소를 선택
                
                # 제품 링크 클릭하여 상세 페이지로 이동
                product_item.click()
                time.sleep(5)
                # exit()
                # 상세 페이지에서 정보 추출
                try:
                    product_name = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, 'h1.DescArticle__TonicName-sc-1df0dck-4'))
                    ).text

                    ingredients = driver.find_elements(By.CSS_SELECTOR, 'li.CharRowUserCheck__ChartTitle-sc-1diyzoj-2')
                    ingredient_list = [ingredient.text for ingredient in ingredients]

                    functionalities = driver.find_elements(By.CSS_SELECTOR, 'li.PropStatRow__PurposeItem-sc-ekdpnr-1')
                    functionality_list = [func.text for func in functionalities]

                    caution = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, 'div.IngestCareRow__IngestCareTd-sc-1w9b4mg-0'))
                    ).text

                    # 결과 출력
                    print(f'제품 랭킹 {rank}')
                    print(f'제품 이름: {product_name}')
                    print('성분:')
                    for ingredient in ingredient_list:
                        print(f'- {ingredient}')
                    print('기능성:')
                    for func in functionality_list:
                        print(f'- {func}')
                    print(f'주의사항: {caution}')

                    with open("data.csv", "a", encoding="utf-8") as f:
                        f.write(f"{nutrient}\t")
                        f.write(f"{product_name}\t")
                        for ingredient in ingredient_list:
                            f.write(f"{ingredient}|")
                        f.write("\t")
                        for func in functionality_list:
                            func = func.strip()
                            f.write(f"{func}|")
                        f.write("\t")
                        caution_list = caution.split("\n")
                        for a in caution_list:
                            f.write(f"{a}|")
                        f.write("\n")


                except Exception as e:
                    print(f'제품 랭킹 {rank}의 정보 추출 중 오류 발생: {e}')

                # 뒤로 가기
                driver.back()
                time.sleep(5)  # 필요에 따라 적절한 시간으로 조정
            else:
                print(f'랭킹 {rank}의 제품이 리스트에 없습니다.')

        # 드랍다운 버튼 클릭하여 영양소 목록으로 돌아가기
        dropdown_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, 'button.DropdownButton__StyledDropdownButton-sc-yyyg8f-0'))
        )
        dropdown_button.click()

finally:
    # WebDriver 종료
    driver.quit()
               