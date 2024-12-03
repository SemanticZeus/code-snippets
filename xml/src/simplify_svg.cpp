//
//  simplify_svg.cpp
//  xml
//
//  Created by Ali Aminian on 12/3/24.
//
#include "simplify_svg.hpp"
#include "ofXml.h"
#include <vector>
#include <map>
#include <iostream>
#include <sstream>
#include <regex>
#include <memory>
#include <string>
#include <algorithm>
#include <iomanip>
#include <cassert>
#include <cmath>
#include <fstream>
#include <ostream>
#include <string>

namespace SIMPLIFY_SVG {

using namespace std::string_literals;

void remove_duplicate_xmlns(std::string& svg)
{
    svg = std::regex_replace(svg, std::regex{R"(xmln.*?=['"].*?['"])"}, "");
    svg = std::regex_replace(svg, std::regex{R"(<svg )"}, "<svg xmlns='http://www.w3.org/2000/svg' ");
}

auto find_pattern(const std::string& s, const std::string& pattern)
{
    std::regex reg{pattern};
    std::smatch matches;
    std::regex_search(s, matches, reg);
    return matches;
}

auto find_pattern_all(const std::string& s, const std::string& pattern)
{
    std::vector<std::smatch> ret;
    std::regex reg{pattern};
    for (std::sregex_iterator it(s.begin(),s.end(),reg),end;it!=end;++it) {
        ret.push_back((*it));
    }
    return ret;
}
auto read_svg_commands(const std::string& d_attrib)
{
    std::vector<std::string> commands;
    enum STATE{start, before_decimal, after_decimal};
    int state = start;
    for (auto c : d_attrib) {
        if (isalpha(c)) {
            commands.push_back(std::string{});
            commands.back() += c;// + " "s;
            state = start;
        } else if (c==',') {
            commands.back() += " "s;
            state = start;
        } else if (c=='-') {
            commands.back() += " -"s;
            state = before_decimal;
        } else if (c=='.' && state == start) {
            commands.back() += " 0.";
            state = after_decimal;
        } else if (c=='.' && state == before_decimal) {
            commands.back() += ".";
            state = after_decimal;
        } else if (isdigit(c) && state == start) {
            commands.back() += " "s + c;
            state = before_decimal;
        } else if (isdigit(c) && state != start) {
            commands.back() += c;
        } else if (c=='.' && state == after_decimal) {
            commands.back() += " "s + c;
            state = after_decimal;
        } else if (c==' ' || c=='\n' || c=='\t') {
            if (state==start) continue;
            commands.back() += " ";
            state = start;
        } else {
            std::cerr << "unknown state" << std::endl;
            std::cerr << d_attrib << std::endl;
        }
    }
    return commands;
}

std::string convert_builder_to_string(const std::vector<std::string> &builder)
{
    size_t size = 0;
    for (auto & b : builder) size+=b.size();
    std::string ret;
    ret.reserve(size);
    for (auto &b : builder) ret+=b;
    return ret;
}

std::string svg_path_add_offset_to_uppercase_letters(std::string path, float xoff, float yoff)
{
    std::string ret;
    std::vector<std::string> ret_builder;
    std::smatch match;
    
    size_t start_index = 0;
    match = find_pattern(path, R"(\sd=['"]([\w\W]*?)['"])");
    ret_builder.push_back(std::string{path.begin()+start_index, path.begin()+match.position(1)});
    start_index += match.position(1)+match[1].length();
    
    for (auto &command : read_svg_commands(match[1].str())) {
        std::ostringstream oss;
        std::istringstream iss{command};
        oss << std::fixed << std::setprecision(2);
        char command_type;
        iss >> command_type;
        if (!std::isalpha(command_type)) {
            std::cout << "Error invalid svg path" << std::endl;
            continue;
        }
        oss << command_type + " "s;
        while (iss) {
            if (command_type == 'z' || command_type == 'Z') {
                oss << " ";
                break;
            } else if (command_type == 'V') {
                std::string f; iss >> f;
                if (iss.fail()) break;
                oss << (std::stof(f)+yoff) << " ";
            } else if (command_type == 'H') {
                std::string f; iss >> f;
                if (iss.fail()) break;
                oss << (std::stof(f)+xoff) << " ";
            } else if (std::isupper(command_type)) {
                std::string f1,f2;
                iss >> f1 >> f2;
                if (iss.fail()) break;
                oss << std::stof(f1)+xoff << "," << std::stof(f2)+yoff << " ";
            }
            else if (command_type == 'v' || command_type == 'h') {
                std::string f;
                iss >> f;
                if (iss.fail()) break;
                oss << f << " ";
            } else {
                std::string f1,f2;
                iss >> f1 >> f2;
                if (iss.fail()) break;
                oss << std::stof(f1) << "," << std::stof(f2) << " ";
            }
        }
        ret_builder.push_back(oss.str());
    }
    ret_builder.push_back(std::string{path.begin()+start_index, path.end()});
    return convert_builder_to_string(ret_builder);
}


bool is_shape_element(ofXml& n)
{
    static const std::vector<std::string> shape_elements = {
        "path",
        "circle",
        "ellipse",
        "rect",
        "line",
        "polygon",
        "polyline",
        "use"
    };
    for (auto &s : shape_elements)
        if (n.getName()== s) return true;
    return false;
}

void modify_xml(ofXml& host, auto f)
{
    f(host);
    for (auto &node : host.getChildren()) modify_xml(node, f);
}

void remove_all_newlines_between_quotation(ofXml& xml)
{
    auto f = [](ofXml& n) {
        for (auto &attrib : n.getAttributes()) {
            auto v = attrib.getValue();
            for_each(v.begin(), v.end(), [](auto &ch) { if(ch=='\n' || ch=='\r') ch=' ';});
            n.setAttribute(attrib.getName(), v);
        }
    };
    modify_xml(xml, f);
}

std::string xml2string(ofXml& xml)
{
    auto n = xml.getName();
    if (!isalpha(n[0]) && n[0]!='_') return "";
    std::string svg;
    svg = "<" + n;
    for (auto &a : xml.getAttributes()) {
        svg += " " + a.getName()+"=\""+a.getValue()+"\"";
    }
    std::string sub_nodes;
    for (auto &n : xml.getChildren()) {
        sub_nodes+=xml2string(n) + "\n";
    }
    if (sub_nodes.size()==0&&xml.getValue().size()==0) {
        svg+="/>";
    } else {
        svg+=">\n" + sub_nodes + "\n" + xml.getValue();
        svg+="\n </"+n+">\n";
    }
    return svg;
}

void turn_first_m_in_path_to_uppercase(std::string& svg)
{
    std::regex firstm{R"(\sd=['"](m))"};
    for (std::sregex_iterator it{svg.begin(),svg.end(),firstm},end;it!=end;++it)
        svg[it->position(1)] = 'M';
}

std::map<std::string,std::string> get_all_path_definitions(ofXml& xml)
{
    std::map<std::string,std::string> paths;
    if (xml.getName()!="defs"s) {
        for (auto &n : xml.getChildren()) {
            auto m = get_all_path_definitions(n);
            for (auto &[n,v] : m)
                paths[n]=v;
        }
    } else {
        for (auto &n : xml.getChildren()) {
            auto id = n.getAttribute("id").getValue();
            if (id=="") continue;
            auto p = xml2string(n);
            turn_first_m_in_path_to_uppercase(p);
            paths[id] = p;
        }
    }
    return paths;
}

std::string make_page_flat(ofXml& xml)
{
    static int counter = 0;
    std::string id = "flat-page-"+std::to_string(counter++);
    std::string page = xml2string(xml);
    page = std::regex_replace(page, std::regex{"<g.*?>"}, "");
    page = std::regex_replace(page, std::regex{"</g.*?>"}, "");
    page = "<g id=\""+id+"\">\n" + page + "\n</g>\n";
    return page;
}

std::string convert_use2path(const std::string& use,
                             std::map<std::string,std::string> defs)
{
    std::smatch match;
    std::regex_search(use, match, std::regex{"xlink.*?=\"#(.*?)\""});
    auto id = match[1].str();
    std::regex_search(use, match, std::regex{"x=\"(.*?)\""});
    float xoff = std::stof(match[1]);
    std::regex_search(use, match, std::regex{"y=\"(.*?)\""});
    float yoff = std::stof(match[1]);
    std::string path = svg_path_add_offset_to_uppercase_letters(defs[id], xoff, yoff);
    return path;
}

void replace_substring(std::string& source, const std::string& substring, const std::string& replacement)
{
    auto pos = source.find(substring);
    if (pos==std::string::npos) {
        std::cout << "Could not find substring" << std::endl;
        return;
    }
    source.replace(pos, substring.length(), replacement);
}

std::string convert_all_uses2paths(const std::string& page, std::map<std::string,std::string> defs)
{
    std::string npage = page;
    std::smatch match;
    auto uses = find_pattern_all(page, R"(<use\s+.*?/>)");
    for (auto &u : uses) {
        replace_substring(npage, u.str(), convert_use2path(u.str(), defs));
    }
    return npage;
}

/*this function finds all drawing elements that belong to no page
 and adds them to their own page.
 it also makes pages flat. that means there will be nonested page elements
 like <g><g><pagecontent></g></g>
 */
std::string organize_pages(ofXml& xml,
                           std::map<std::string,std::string> defs,
                           int open_page=0)
{
    std::string pages;
    static int counter = 0;
    if (xml.getName()=="defs") return "";
    for (auto &n : xml.getChildren()) {
        if (n.getName()=="g") {
            if (open_page==1) {
                pages += "\n</g>\n";
                open_page=0;
            }
            std::string page = make_page_flat(n);
            page = convert_all_uses2paths(page,defs);
            pages += page;
        } else if (is_shape_element(n)) {
            if (open_page==0) {
                open_page = 1;
                pages += "\n<g id=\"unknown-page-"+std::to_string(counter++)+"\">\n";
            }
            pages+=xml2string(n) + " \n";
        } else {
            for (auto &x : organize_pages(n,defs,open_page)) pages += x;
        }
    }
    return pages;
}

auto split_string(const std::string&s , char ch)
{
    std::vector<std::string> ret;
    enum STATE {word, delim};
    STATE state = delim;
    for (size_t i=0;i<s.size();i++) {
        if (s[i]!=ch && state==word) ret.back().push_back(s[i]);
        else if (s[i]!=ch && state==delim) {ret.push_back(std::string{}); ret.back().push_back(s[i]); state=word; }
        else if (state == word && s[i]==ch) state = delim;
        else { state = delim; }
    }
    return ret;
}

auto read_styles(const std::string& svg)
{
    std::map<std::string,std::string> id_attributes;
    std::smatch matches;
    std::regex_search(svg, matches, std::regex{R"(<style[\w\W]*?</style>)"});
    auto styles = matches[0].str();
    for (auto m : find_pattern_all(styles, R"(\.([^{}]*?)\{([^{}]*?)\})")) {
        auto id = m[1].str();
        for (auto attrib : split_string(m[2],';')) {
            auto name_value = split_string(attrib,':');
            if (name_value.size()!=2) continue;
            id_attributes[id] += name_value[0] + "=\""s + name_value[1] + "\" "s;
        }
    }
    return id_attributes;
}

std::string inline_style_attributes(const std::string& svg)
{
    std::string ret = svg;
    auto id_attributes = read_styles(svg);
    for (auto &[id,attributes] : id_attributes) {
        auto class_name = "class=\"" + id + "\"";
        ret = std::regex_replace(ret, std::regex{class_name}, id_attributes[id]);
    }
    ret = std::regex_replace(ret, std::regex{R"(<style[\s\S]*?</style>)"}, "");
    return ret;
}

auto find_bounding_box(const std::string& svg)
{
    float minx = std::numeric_limits<float>::max();
    float miny = std::numeric_limits<float>::max();
    float maxx = std::numeric_limits<float>::min();
    float maxy = std::numeric_limits<float>::min();
    float x=-2000,y=-2000;
    for (auto &m : find_pattern_all(svg, R"(<path[\s\S]*?/>)")) {
        std::string d_attrib = find_pattern(m.str(), R"(\sd=['"]([\s\S]*?)['"])")[1].str();
        auto commands = read_svg_commands(d_attrib);
        for (auto &command : commands) {
            std::for_each(command.begin(),command.end(), [](char& ch){ return ch==',' ? ' ' : ch;});
            char ch;
            std::istringstream iss(command);
            iss >> ch;
            for (size_t i=0;!iss.fail();i++) {
                std::string s;
                iss >> s;
                if (iss.fail()) break;
                float number = std::stof(s);
                if (ch == 'H') x = number;
                else if (ch == 'V') y = number;
                else if (ch == 'h') x += number;
                else if (ch == 'v') y += number;
                else if (isupper(ch) && i%2==0) x = number;
                else if (isupper(ch) && i%2==1) y = number;
                else if (islower(ch) && i%2==0) x += number;
                else if (islower(ch) && i%2==1) y += number;
                if (x==-2000 || y ==-2000) continue;
                if (x>maxx) maxx = x;
                if (y>maxy) maxy = y;
                if (x<minx) minx = x;
                if (y<miny) miny = y;
            }
        }
    }
    return std::tuple{minx,miny,maxx,maxy};
}

std::string fix_header(const std::string& svg)
{
    auto [minx,miny,maxx,maxy] = find_bounding_box(svg);
    std::string svg_header = R"(<svg version='1.1' xmlns='http://www.w3.org/2000/svg' width=')"s;
    svg_header += std::to_string(maxx-minx) + "pt' height='"s + std::to_string(maxy-miny);
    svg_header += "pt'>\n"s;
    auto ret = std::regex_replace(svg, std::regex{R"(<svg[\w\W]*?>)"}, svg_header);
    return ret;
}

std::string tr(const std::string& s, int n)
{
    if (s.size()<n) return s;
    auto ss = s.substr(0,n/2) + "  ....  " + s.substr(s.size()-n/2, s.size());
    return ss;
}

std::string print(const ofXml& host, std::string indent="")
{
    std::ostringstream oss;
    for (auto &node : host.getChildren()) {
        oss << indent << "<" + node.getName() + ">" << std::endl;
        for (auto attrib : node.getAttributes()) {
            std::string n = attrib.getName();
            std::string v = attrib.getValue();
            oss << indent + "   " + tr(n,30) << ": " << tr(v,30) << std::endl;
        }
        
        oss << print(node, indent+"      ");
        oss << indent << "</" + node.getName() + ">" << std::endl;
    }
    return oss.str();
}

std::string remove_all_pages_and_defs(ofXml& xml)
{
    std::string svg;
    char first_letter = xml.getName()[0];
    if(isalpha(first_letter) || first_letter=='_') {
        svg = "<" + xml.getName();
        for (auto &a : xml.getAttributes()) {
            svg += " " + a.getName()+"=\""+a.getValue()+"\"";
        }
        svg+=">\n";
        svg+=xml.getValue()+ "\n";
    }
    for (auto &n : xml.getChildren()) {
        if (is_shape_element(n) || n.getName()=="g"s) continue;
        if (n.getName()=="defs"s) continue;
        svg+=remove_all_pages_and_defs(n);
    }
    if(isalpha(first_letter) || first_letter=='_') {
        svg+="\n</"+xml.getName()+">\n";
    }
    return svg;
}

std::string merge_all_pages(const std::string& svg)
{
    std::string ret;
    std::vector<std::string> pages;
    pages.push_back("<g id='merged_page'>\n");
    ret = std::regex_replace(svg, std::regex{R"(<g[\s\S]*?</g>)"}, "");
    size_t size = 50 + ret.size();
    for (auto &m : find_pattern_all(svg, R"(<g[\s\S]*?</g>)")) {
        std::string page = std::regex_replace(m.str(), std::regex{R"(<g[^>]*?>)"}, "");
        page = std::regex_replace(page, std::regex{R"(</g>)"}, "");
        size+=page.size();
        pages.push_back(std::move(page));
    }
    pages.push_back("</g>\n</svg>");
    ret = std::regex_replace(ret, std::regex{R"(</svg>)"}, "");
    ret.reserve(size);
    for (auto &page : pages) {
        ret += "\n"s + page;
    }
    return ret;
}

std::string remove_extra_white_spaces(const std::string &svg)
{
    std::string ret = svg;
    ret = std::regex_replace(ret, std::regex{R"([\t ]+)"}, " ");
    ret = std::regex_replace(ret, std::regex{R"([\n]+)"}, "\n");
    return ret;
}

std::string simplify_svg(const std::string& filename)
{
    ofXml xml;
    xml.load(filename);
    remove_all_newlines_between_quotation(xml);
    auto defs = get_all_path_definitions(xml);
    auto organized_pages = organize_pages(xml,defs);
    std::string svg = remove_all_pages_and_defs(xml);
    svg = std::regex_replace(svg, std::regex{"</svg.*>"}, organized_pages+"</svg>");
    svg = std::regex_replace(svg, std::regex{"\n+"}, "\n");
    svg = merge_all_pages(svg);
    svg = inline_style_attributes(svg);
    svg = remove_extra_white_spaces(svg);
    svg = fix_header(svg);
    svg = merge_all_pages(svg);
    return svg;
}

}
