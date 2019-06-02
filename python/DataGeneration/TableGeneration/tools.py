import traceback


from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver import Firefox
from selenium.webdriver import PhantomJS

from selenium.webdriver.firefox.options import Options
from PIL import Image
from io import BytesIO
import warnings
import json
import time
def warn(*args, **kwargs):
    pass

warnings.warn = warn



def html_to_img(driver,html_content,id_count,max_height,max_width):
    # opts = Options()
    # opts.set_headless()
    # assert opts.headless
    # #driver=PhantomJS()
    # driver = Firefox(options=opts)
    counter=1
    while(True):
        try:
            driver.get("data:text/html;charset=utf-8," + html_content)
            #driver.execute_script("document.write('{}')".format(json.dumps(htmlcode)))

            element = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.ID, '0')))

            #WebDriverWait(driver, 1).until(EC.visibility_of(element))


            bboxes=[]
            for id in range(id_count):
                #e = driver.find_element_by_id(str(id))
                e = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.ID, str(id))))
                txt=e.text.strip()
                lentext=len(txt)
                loc = e.location
                size_ = e.size
                xmin = loc['x']
                ymin = loc['y']
                xmax = int(size_['width'] + xmin)
                ymax = int(size_['height'] + ymin)
                bboxes.append([lentext,txt,xmin,ymin,xmax,ymax])
                # cv2.rectangle(im,(xmin,ymin),(xmax,ymax),(0,0,255),2)

            png = driver.get_screenshot_as_png()

            im = Image.open(BytesIO(png))
            width,height=im.size

            im = im.crop((0,0, max_width, max_height))

            #im.save(outimgpath,dpi=(600,600))
            return im,bboxes
        except Exception as e:
            counter+=1
            if(counter==10):
                raise e
                #print('\nraising exception')
                #traceback.print_exc()
                #raise Exception("Error occurred at line 68 inside tools.py")

            continue