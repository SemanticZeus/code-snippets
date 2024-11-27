import xml.etree.ElementTree as ET
import re
import string

def replace_group_references(svg_file, output_file):
    tree = ET.parse(svg_file)
    root = tree.getroot()
    namespaces = {'svg': 'http://www.w3.org/2000/svg'}
    ET.register_namespace('', namespaces['svg'])
    defs = root.find('svg:defs', namespaces)
    if defs is None:
        print("No <defs> section found in the SVG.")
        tree.write(output_file, encoding='utf-8', xml_declaration=True)
        return
    if root.find('svg:xmlns',namespaces) == None:
        print ("adding xmlns to svg")
    path_dict = {}
    for elem in defs:
        if elem.tag.endswith('path') and 'id' in elem.attrib:
            path_dict[elem.attrib['id']] = elem
    idcount = 0
    for g in root.findall('.//svg:g', namespaces):
        for use in g.findall('svg:use', namespaces):
            href = use.attrib.get('{http://www.w3.org/1999/xlink}href')
            if href and href.startswith('#'):
                path_id = href[1:]
                if path_id in path_dict:
                    path_elem = path_dict[path_id]
                    x = use.attrib.get('x')
                    y = use.attrib.get('y')
                    path_elem.set('xxx', x)
                    path_elem.set('yyy', y)
                    path_elem.set('id', 'id{}'.format(idcount))
                    idcount+=1
                    g.append(path_elem)
                    g.remove(use)
    tree.write(output_file, encoding='utf-8', xml_declaration=True)

def f(x,y):
    def ff(s):
        s = s.group()
        e = s.split()
        r = f"{e[0]} "
        if e[0] in "VH":
            if e[0] == "V":
                e[1] = float(e[1])+y
            elif e[0] == "H":
                e[1] = float(e[1])+x
            r = r + " {:.1f} ".format(e[1])
        elif e[0] in string.ascii_uppercase:
            for i in range(1,len(e), 2):
                e[i] = float(e[i])+x
                e[i+1] = float(e[i+1])+y
            for i in range(1, len(e), 2):
                r = r + " {:.1f},{:.1f} ".format(float(e[i]),float(e[i+1]))
        elif e[0] in "vh":
            if e[0] == "v":
                e[1] = float(e[1])+y
            elif e[0] == "h":
                e[1] = float(e[1])+x
            r = r + " {:.1f} ".format(e[1])
        elif e[0] in string.ascii_lowercase:
            for i in range(1,len(e), 2):
                e[i] = float(e[i])+x
                e[i+1] = float(e[i+1])+y
            for i in range(1, len(e), 2):
                r = r + " {:.1f},{:.1f} ".format(float(e[i]),float(e[i+1]))

        else:
            return s
        return r
    return ff



def convert_svg_paths(svg_file, output_file):
    tree = ET.parse(svg_file)
    root = tree.getroot()

    namespaces = {'svg': 'http://www.w3.org/2000/svg'}
    ET.register_namespace('', namespaces['svg'])
    namespaces = {'svg': 'http://www.w3.org/2000/svg'}
    ET.register_namespace('', namespaces['svg'])
    # Iterate over all <path> elements and convert them to the new format
    for path in root.findall('.//svg:path', namespaces):
        # Update the path id and add data-name attribute
        path_id = path.attrib.get('id', 'unknown')
        new_id = path_id.replace('g', 'g3-')  # Example transformation for new id
        path.attrib['id'] = new_id
        x = float(path.attrib.get('xxx') or 0)
        y = float(path.attrib.get('yyy') or 0)
        # Format the 'd' attribute for consistency with spacing and comma separation
        d_attribute = path.attrib.get('d', '')
        #print(d_attribute)
        d_attribute = re.sub(r',', r' ',d_attribute)
        d_attribute = re.sub(r'([a-zA-Z])', r' \1 ', d_attribute)
        d_attribute = re.sub(r'(?<=\d)(-)', r' \1', d_attribute)
        for c in string.ascii_uppercase:
            d_attribute = re.sub(r'{}[^a-zA-Z]*'.format(c), f(x,y), d_attribute)
        for c in string.ascii_lowercase:
            d_attribute = re.sub(r'{}[^a-zA-Z]*'.format(c), f(0,0), d_attribute)


        d_attribute = re.sub(r'\s+', ' ', d_attribute).strip()  # Normalize spaces
        path.attrib['d'] = d_attribute
    
    # Write the modified SVG to the output file
    tree.write(output_file, encoding='utf-8', xml_declaration=True)


import sys
input_svg = sys.argv[1]
if input_svg == None:
    print("error: usage {argv[0]} <svg_file>")
    exit()
output_svg = 'output.svg'
replace_group_references(input_svg, output_svg)
convert_svg_paths(output_svg, output_svg)
with open(output_svg,'r') as f:
    data = f.read()
    data = re.sub(r'data-name=".*?"','',data)
    data = re.sub(r'<defs>[\w\W]*</defs>','',data)

with open(output_svg,'w') as f:
    f.write(data)
print(f"Modified SVG saved as {output_svg}")

