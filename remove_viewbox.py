import re
import os

re_viewbox = r"viewBox.*?\".*?\""
re_style = r"<style type=\"text/css\">(?P<styles>[\w\W]*?)</style>"
re_style_id_color = r"\.(?P<id>.*?){(?P<attribute>.*?);}"


os.system("cp ./backup/* ./")
files = [f for f in os.listdir('.') if f.endswith(".svg")] 
for f in files:
    with open(f, 'r') as fh:
        text = fh.read()
        text = re.sub(re_viewbox,"",text)
        styles = re.findall(re_style,text)
        if len(styles) == 0:
            continue
        print(f)
        text = re.sub(re_style,"",text)
        styles = re.findall(re_style_id_color, styles[0])
        print(f'styles = {styles}')
        for name,attributes in styles:
            attributes = attributes.split(';')
            s = ''
            print(f'attributes = {attributes}')
            for a in attributes:
                n,v = a.split(':')
                s += f' {n}="{v}"'
            text = re.sub(f'class="{name}"',s,text)
    with open(f,'w') as fh:
        fh.write(text)


