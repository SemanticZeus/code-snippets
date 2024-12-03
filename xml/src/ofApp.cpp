#include "ofApp.h"
#include "ofxSvg.h"
#include "simplify_svg.hpp"
#include <vector>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;


auto find_files(const std::string& root, const std::string& suffix)
{
    std::vector<std::string> files;
    for (auto &e : fs::directory_iterator(root)) {
        if (e.is_regular_file() && e.path().string().ends_with(suffix))
            files.push_back(e.path().string());
    }
    return files;
}

std::vector<ofxSvg> svgs;
ofxSvg maxwell;
float x_step = 150;
float y_step = 150;

auto svg_get_size(const ofxSvg& svg)
{
    float minx = std::numeric_limits<float>::max();
    float miny = std::numeric_limits<float>::max();
    float maxx = std::numeric_limits<float>::min();
    float maxy = std::numeric_limits<float>::min();
    for (auto path: svg.getPaths()) {
        path.setStrokeWidth(1);
        auto outline = path.getOutline();
        for (auto o : outline) {
            auto v = o.getResampledByCount(100).getVertices();
            for (auto p : v) {
                if (p.x<minx) minx = p.x;
                if (p.x>maxx) maxx = p.x;
                if (p.y<miny) miny = p.y;
                if (p.y>maxy) maxy = p.y;
            }
        }
    }
    return make_tuple(minx,miny,maxx,maxy);
}


void draw_svg(ofxSvg& svg, float x, float y, float width, float height)
{
    
    auto [minx,miny,maxx,maxy] = svg_get_size(svg);
    float w = maxx-minx;
    float h = maxy-miny;
    float s = (width-6)/w;
    if (s*h>(height-6)) s = (height-6)/h;
    for (auto p : svg.getPaths()) {
        ofPushMatrix();
        ofSetColor(p.getFillColor());
        p.translate(glm::vec2{-minx,-miny});
        p.scale(s,s);
        p.setArcResolution(500);
        //ofTranslate(x,y);
        p.draw(x+3,y+3);
        ofPopMatrix();
    }
    ofPushMatrix();
    //svg.draw();
    ofSetColor(0,0,0);
    ofNoFill();
    ofTranslate(x, y);
    ofDrawRectangle(0, 0, width, height);
    ofPopMatrix();
}

//--------------------------------------------------------------
void ofApp::setup(){
auto root = "/Users/aliaminian/manimgl/manim_pi_creatures/PiCreature/"s;
    for (auto &f : find_files(root, ".svg")) {
        //if (f.find("miner")!=std::string::npos) continue;
        //if (f.find("sassy")==std::string::npos) continue;
        svgs.push_back(ofxSvg{});
        svgs.back().loadFromString(simplify_svg(f));
    }
    for (auto &f : find_files("/Users/aliaminian/codesnippets/svg", ".svg")) {
        //if (f.find("miner")!=std::string::npos) continue;
        //if (f.find("sassy")==std::string::npos) continue;
        svgs.push_back(ofxSvg{});
        svgs.back().loadFromString(simplify_svg(f));
    }
    svgs.push_back(ofxSvg{});
    svgs.back().loadFromString(simplify_svg("/Users/aliaminian/tiger.svg"));
    maxwell.loadFromString(simplify_svg("/Users/aliaminian/files/maxwell.svg"));
}

//--------------------------------------------------------------
void ofApp::update(){

}

//--------------------------------------------------------------
void ofApp::draw(){
    float x = 50;
    float y = 50;
    ofBackground(255);
    draw_svg(maxwell,20, 20, ofGetWindowWidth()-40, ofGetWindowHeight()-40);
    for (auto &svg : svgs) {
        draw_svg(svg, x, y, x_step, y_step);
        x+= x_step;
        if (x>ofGetWidth()-x_step) {x = 50; y += y_step;}
    }
}

//--------------------------------------------------------------
void ofApp::exit(){

}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){

}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseScrolled(int x, int y, float scrollX, float scrollY){

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){

}
