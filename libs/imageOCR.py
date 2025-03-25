
from tqdm import tqdm
from re import sub
from paddleocr import PaddleOCR
from pathlib import Path
import os
import re

ocr = PaddleOCR(lang='es',use_angle_cls=False,show_log = False)
def filtradoNombre(nombre):
    x = sub(r"\.avi$", "", nombre)
    x = sub(r"\.mp4$", "", x)
    x = sub(r"\.mkv$", "", x)
    x = sub(r"\.ts$", "", x)
    x = sub(r"\.rmvb", "", x)
    x = x.strip()
    return x

def reemplazar(test_str=''):
    regex = r"[^A-Za-zÑñÁáÉéÍíÓóÚúÜü?! ,.;:¡¿]"
    subst = ""

    # You can manually specify the number of replacements by changing the 4th argument
    result = re.sub(regex, subst, test_str, 0)
    return result

def ocrPaddle ():
    list_list ={}
    current_folder_path, current_folder_name = os.path.split(os.getcwd())
    current_directory = Path(Path.cwd())
    images_dir = Path(f'{current_directory}/')
    if not images_dir.exists():
        images_dir.mkdir()
        print('Images folder is empty.')
        exit()
    imagesList = Path(images_dir).rglob('*.jpeg')
    images=[]
    for image in imagesList:
        images.append(image)
    line = 1
    for image in tqdm(images,desc=current_folder_name):
        result = ocr.ocr(str(image.absolute()),det=True,rec=True,cls=False)
        # print(result)
        texts = [line[1][0] for line in result[0]] if result and result[0]  else []
        text_content = ' '.join(texts)
        imgname = image.name
        start_hour = imgname.split('_')[0][:2]
        start_min = imgname.split('_')[1][:2]
        start_sec = imgname.split('_')[2][:2]
        start_micro = imgname.split('_')[3][:3]
        # end_hour = imgname.split('__')[1].split('_')[0][:2]
        # end_min = imgname.split('__')[1].split('_')[1][:2]
        # end_sec = imgname.split('__')[1].split('_')[2][:2]
        # end_micro = imgname.split('__')[1].split('_')[3][:3]
        # Format start time
        start_time = f'{start_hour}:{start_min}:{start_sec},{start_micro}'
        # Format end time
        # end_time = f'{end_hour}:{end_min}:{end_sec},{end_micro}'
        # Append the line to srt file
        list_list[line] = [
            f'{line}\n',
            f'{start_time} --> {start_time}\n',
            f'{reemplazar(text_content)}\n\n',
                ''
            ]
        line+=1
    srt_file = open(
            Path(f'{str(current_directory).replace(current_folder_name,"Subtitles")}/{current_folder_name}.srt'), 'a', encoding='utf-8')
    for i in sorted(list_list):
        srt_file.writelines(list_list[i])
    srt_file.close()
def main():
    ocrPaddle()

if __name__ == "__main__":
    main()