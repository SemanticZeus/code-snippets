//
//  mobject.h
//  PI_Animation
//
//  Created by Ali Aminian on 11/23/24.
//

#include "ofApp.h"
#include "ofxSVG.h"
#include <boost/type_index.hpp>

template <typename T>
void printType(const T& var) {
    std::cout << "Type: " << boost::typeindex::type_id_with_cvr<T>().pretty_name() << '\n';
}

#include <regex>
namespace fs = std::filesystem;

std::vector<std::string> find_files(std::string root, const std::string &pattern, int depth=1);

using SubPath = std::vector<std::vector<ofPoint>>;
using Path = std::vector<SubPath>;

class Frame2 {
public:
    ofPath render();
    std::vector<int> handles;
    std::vector<std::function<void(int)>> functions;
};

class MObject {
public:
    MObject(const std::string& filename);
    MObject() {};
    void draw() const;
    void draw2() const;
    void set_color(ofColor c);
    Path paths;
    std::vector<ofColor> fill_colors;
    std::vector<ofColor> stroke_colors;
    std::vector<int> stroke_width;
    std::vector<int> filled;
    float scale;
    float x=0,y=0;
    float w=0,h=0;
    int outer_path=-1;
    mutable ofxSvg svg;
    mutable std::vector<ofPath> of_paths;
};

class PiCreature : public MObject {
    PiCreature(const std::string& filename) : MObject{filename} {}
    constexpr const static int left_eye = 1;
    constexpr static int right_eye = 2;
    constexpr static int left_pupil = 3;
    constexpr static int right_pupil = 4;
    void move_pupil(double x, double y);
};

using Frame = std::vector<MObject>;
extern std::vector<Frame> frames;

MObject translate(const MObject& m, std::function<ofPoint(ofPoint)> f);

MObject interpolate2mobjects(
                             const MObject& m1,
                             const MObject& m2, double tran_step);
MObject interpolate2mobjects(
                             const MObject& m1,
                             const MObject& m2, double tran_step,std::vector<int>points);

std::vector<Frame> interpolate2frames(
                                      const std::vector<MObject>& f1,
                                      const std::vector<MObject>& f2,
                                      double tran_step);

void draw_frame(int i);
void scale_frame(Frame& frame, double scale);


std::tuple<float,float,float,float> get_frame_bounding_box(const Frame& frame);
void change_alpha(Frame& frame, int alpha);
std::vector<int> findNearestPoints(const std::vector<ofPoint>& vectorA, const std::vector<ofPoint>& vectorB,float s1, float s2, float x1, float y1, float x2, float y2);
