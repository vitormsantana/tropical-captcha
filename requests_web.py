import os
import requests

def download_captcha(url, n=1, path=".", secure=False, ext=".jpeg"):
    # Modify the path to the desired folder
    path = os.path.join(path, "labeled_testSet")
    os.makedirs(path, exist_ok=True)

    # Replace known captchas
    if url == "tjrs":
        url = "http://www.tjrs.jus.br/site_php/consulta/human_check/humancheck_showcode.php"
    elif url == "tjmg":
        url = "http://www4.tjmg.jus.br/juridico/sf/captcha.svl"
    elif url == "tjrj":
        url = "http://www4.tjrj.jus.br/consultaProcessoWebV2/captcha"
    elif url == "trt":
        url = "https://consultapje.trt1.jus.br/consultaprocessual/seam/resource/captcha"
    elif url == "rfb":
        url = "http://www.receita.fazenda.gov.br/pessoajuridica/cnpj/cnpjreva/captcha/gerarCaptcha.asp"

    # Iterate over downloads
    out = []
    for i in range(n):
        out.append(download_captcha_(url, path, secure, ext, i))

    return out

def download_captcha_(url, path, secure, ext, index):
    # Send GET request
    r = requests.get(url, verify=secure)

    # Get file extension
    ct = r.headers.get("content-type")
    ext = ext if ct is None else "." + ct.split("/")[-1]

    # Save captcha to disk with unique file name
    file_name = f"captcha_{index:03d}{ext}"
    file_path = os.path.join(path, file_name)
    with open(file_path, "wb") as f:
        f.write(r.content)

    return file_path

# Example usage: Download 500 captchas from RFB
num_captchas = 200
file_paths = download_captcha("rfb", n=num_captchas, path="C:/Users/visantana/Documents/tropical-captcha")
print(f"Downloaded {num_captchas} files:", file_paths)


