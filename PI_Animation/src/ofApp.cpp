#include <filesystem>
#include "mobject.h"
#include "music.h"

const int nx = 10;
const int ny = 5;

std::string pi_svg_path = "/Users/aliaminian/codesnippets/svg/Pi-symbol.svg";

std::vector<Frame> animation_3b1b_demo(Frame starting_frame, float width, float height)
{
    std::vector<Frame> frames;
    auto [minx,miny,maxx,maxy] = get_frame_bounding_box(starting_frame);
    float frame_w = maxx-minx;
    float frame_h = maxy-miny;
    
    scale_frame(starting_frame, TWO_PI/frame_h);
    frame_w *=TWO_PI/frame_h;
    frame_h = TWO_PI;
    float ns = width/frame_w;
    if (frame_h*ns>height) ns = height/frame_h;
    
    for (int i=0;i<10;i++) {
        auto f = starting_frame;
        frames.push_back(std::move(f));
    }
    
    for (int i=1;i<100;i++) {
        Frame f = starting_frame;
        scale_frame(f, ns*i/100.0);
        frames.push_back(std::move(f));
    }
    
    for (int i=0;i<100;i++) frames.push_back(frames.back());
    
    auto new_frame = starting_frame;
    ns = TWO_PI/frame_h;
    scale_frame(new_frame, ns);
    auto last_frame = frames.back();
    for (auto &m : new_frame) m = translate(m, [](const auto& p) {return glm::vec2{exp(p.x)*cos(p.y), exp(p.x)*sin(p.y)};});
    for (auto &f : interpolate2frames(last_frame, new_frame, .01)) {
        frames.push_back(std::move(f));
    }
    for (int i=0;i<100;i++) frames.push_back(frames.back());
    
    last_frame = frames.back();
    new_frame = starting_frame;
    ns = TWO_PI/frame_h;
    scale_frame(new_frame, ns);
    auto f1 = [](const auto& p) { return glm::vec2{exp(p.x*.8)*cos(p.y), exp(p.x*.8)*sin(p.y)};};
    for (auto &m : new_frame) m = translate(m, f1);
    auto b = get_frame_bounding_box(new_frame);
    ns = TWO_PI/(std::get<3>(b)-std::get<1>(b));
    scale_frame(new_frame, ns);
    for (auto &m : new_frame) m = translate(m, [](const auto& p) { return glm::vec2{p.x+.5*sin(p.y), p.y+.5*sin(p.x)};});
    auto last_box = get_frame_bounding_box(last_frame);
    auto new_box = get_frame_bounding_box(new_frame);
    
    float last_w = std::get<2>(last_box)-std::get<0>(last_box);
    float last_h = std::get<3>(last_box)-std::get<1>(last_box);
    float new_w = std::get<2>(new_box)-std::get<0>(new_box);
    float new_h = std::get<3>(new_box)-std::get<1>(new_box);
    std::cout << "last_w = " << last_w << ", new_w = " << new_w << std::endl;
    ns = last_w/new_w;
    if (ns*new_h>last_h) ns = last_h/new_h;
    scale_frame(new_frame, ns);
    for (auto &f : interpolate2frames(last_frame, new_frame, .01)) frames.push_back(std::move(f));
    for (int i=0;i<100;i++) frames.push_back(frames.back());
    return frames;
}

void init_scene(int nx, int ny)
{
    MObject pi(pi_svg_path);
    float max_height = TWO_PI;
    float y_step = (max_height)/ny;
    float x_step = y_step/pi.h * pi.w;
    Frame frame;
    for (int i=0;i<nx;i++) {
        for (int j=0;j<ny;j++) {
            auto npi = pi;
            npi.set_color(ofColor{ofRandom(255), ofRandom(255), ofRandom(255)});
            npi.x = x_step*(i-nx/2);
            npi.y = y_step*(j-ny/2);
            float s = x_step/npi.w;
            if (s*npi.h>y_step) s = y_step/npi.h;
            npi.scale = s*.7;
            frame.push_back(npi);
        }
    }
    for (auto &f: animation_3b1b_demo(frame, ofGetWidth(), ofGetHeight()))
        frames.push_back(std::move(f));
}

void init_pi_creatures(int nx, int ny, float width, float height)
{
    std::string pattern{".*PiCreatures.*\\.svg$"};
    std::string root{"/Users/aliaminian/manimgl/manim_pi_creatures/PiCreature"};
    auto pi_svgs = find_files(root, pattern,0);
    Frame frame;
    float x_step = width/ny;
    float y_step = height/ny;
    for (int i=0;i<nx;i++) {
        for (int j=0;j<ny;j++) {
            auto file = pi_svgs[(i*nx+j)%pi_svgs.size()];
            MObject m{file};
            m.x = x_step*(i-nx/2);
            m.y = y_step*(j-ny/2);
            float s = x_step/m.w;
            if (s*m.h>y_step) s = y_step/m.h;
            m.scale = s*.9;
            frame.push_back(std::move(m));
        }
    }
    for (auto &f: animation_3b1b_demo(frame, ofGetWidth(), ofGetHeight()))
        frames.push_back(std::move(f));
}

void init_pi_creatures_transistion(float width, float height)
{
    std::string pattern{".*PiCreatures.*\\.svg$"};
    std::string root{"/Users/aliaminian/manimgl/manim_pi_creatures/PiCreature"};
    auto pi_svgs = find_files(root, pattern,0);
    std::vector<MObject> pis;
    for (auto &file : pi_svgs) {
        MObject pi{file};
        if (pi.paths.size()!=6) continue;
        std::cout << file << std::endl;
        float s = ofGetWidth()/2/pi.w;
        if (s*pi.h>ofGetHeight()/2) s = ofGetHeight()/2/pi.h;
        pi.scale = s;
        pi.x = -ofGetWidth()/2 + 50;
        pi.y = -ofGetHeight()/2 + 50;
        pis.push_back(std::move(pi));
    }
    auto &pi0 = pis[0];
    auto &pi1 = pis[11];
    Frame frame1,frame2;
    frame1.push_back(pi0);
    for (int i=1;i<5;i++) {
        Frame frame1,frame2;
        frame1.push_back(pis[i-1]);
        frame2.push_back(pis[i]);
        for (int i=0;i<50;i++) frames.push_back(frame1);
        int num_steps = 20;
        for (auto &frame : interpolate2frames(frame1, frame2, .01)) {
            frames.push_back(std::move(frame));
        }
        for (int i=0;i<50;i++) frames.push_back(frame2);
    }
}

//--------------------------------------------------------------
void ofApp::setup() {
    //init_scene(20,10);
    //init_pi_creatures(20, 10, ofGetWidth(), ofGetHeight());
    init_pi_creatures_transistion(ofGetWidth(), ofGetHeight());
    ofEnableAlphaBlending();
    std::cout << "frames.size = " << frames.size() << std::endl;
}

//--------------------------------------------------------------
void ofApp::update() {
    
}

int frame_index = 0;
int v = 1;
//--------------------------------------------------------------
void ofApp::draw() {
    if (frames.size()==0) return;
    ofPushStyle();      // Save current style
    ofPushMatrix();
    ofBackground(255, 255, 255);
    ofTranslate(ofGetWidth()/2, ofGetHeight()/2);
    draw_frame(frame_index);
    ofPopMatrix();
    ofPopStyle();
    ofSetColor(0,0,0);

    if (frames.size()==1) return;
    
    if (frame_index==frames.size()-1) v = -1;
    else if (frame_index==0) v = 1;
    frame_index+=v;
}

//--------------------------------------------------------------
void ofApp::exit() {
    
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key) {

}

//--------------------------------------------------------------
void ofApp::keyReleased(int key) {

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ) {
    
}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button) {
    
}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button) {
    
}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button) {
    
}

//--------------------------------------------------------------
void ofApp::mouseScrolled(int x, int y, float scrollX, float scrollY) {
    
}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y) {
    
}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y) {
    
}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h) {
}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg) {
    
}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo) {
    
}
