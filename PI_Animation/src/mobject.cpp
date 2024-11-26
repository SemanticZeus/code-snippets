//
//  mobject.cpp
//  PI_Animation
//
//  Created by Ali Aminian on 11/23/24.
//
#include "mobject.h"
#include <regex>
#include <format>
#include <ranges>

namespace fs = std::filesystem;

std::map<int, MObject> mobjects;

float computeSignedArea(const std::vector<ofPoint>& points) {
    float area = 0;
    for (size_t i = 0; i < points.size(); ++i) {
        const ofPoint& p1 = points[i];
        const ofPoint& p2 = points[(i + 1) % points.size()];
        area += (p1.x * p2.y) - (p2.x * p1.y);
    }
    return area / 2.0f;
}

bool isClockwise(const std::vector<ofPoint>& points) {
    return computeSignedArea(points) < 0; // Negative area indicates CW
}

std::vector<ofPoint> correctWindingOrder(std::vector<ofPoint>& points) {
    if (isClockwise(points)) {
        std::reverse(points.begin(), points.end());
    }
    return points;
}


auto generate_path(const MObject &m)
{
    std::vector<ofPath> paths;
    //path.setMode(ofPath::POLYLINES);
    auto f = [&m](auto i) {
        ofPath path;
        auto &paths = m.paths;
        if (i<0 || i>=paths.size()) return path;
        for (auto &outline : paths[i]) {
            path.newSubPath();
            path.moveTo(outline[0]);
            path.setFilled(m.filled[i]);
            path.setColor(m.stroke_colors[i]);
            path.setFillColor(m.fill_colors[i]);
            path.setStrokeWidth(m.stroke_width[i]);
            for (auto &p : outline) path.lineTo(p);
        }
        path.close();
        return path;
    };
    for (size_t i=0;i<m.paths.size();i++) {
        paths.emplace_back(f(i));
    }
    return paths;
}

auto get_bounding_box(const std::vector<ofPoint>& path)
{
    float minx = std::numeric_limits<float>::max();
    float miny = std::numeric_limits<float>::max();
    float maxx = std::numeric_limits<float>::min();
    float maxy = std::numeric_limits<float>::min();
    for (const auto & v : path) {
        minx = std::min(minx,v.x);
        miny = std::min(miny,v.y);
        maxx = std::max(maxx,v.x);
        maxy = std::max(maxy,v.y);
    }
    return std::tuple{minx,miny,maxx,maxy};
}

auto get_bounding_box(const SubPath& path)
{
    float minx = std::numeric_limits<float>::max();
    float miny = std::numeric_limits<float>::max();
    float maxx = std::numeric_limits<float>::min();
    float maxy = std::numeric_limits<float>::min();
    for (const auto & v : path) {
        auto box = get_bounding_box(v);
        minx = std::min(minx,std::get<0>(box));
        miny = std::min(miny,std::get<1>(box));
        maxx = std::max(maxx,std::get<2>(box));
        maxy = std::max(maxy,std::get<3>(box));
    }
    return std::tuple{minx,miny,maxx,maxy};
}


auto get_bounding_box(const MObject& m)
{
    float minx = std::numeric_limits<float>::max();
    float miny = std::numeric_limits<float>::max();
    float maxx = std::numeric_limits<float>::min();
    float maxy = std::numeric_limits<float>::min();
    for (const auto & path: m.paths) {
        for (auto &subpath : path) {
            auto box = get_bounding_box(subpath);
            minx = std::min(std::get<0>(box), minx);
            miny = std::min(std::get<1>(box), miny);
            maxx = std::max(std::get<2>(box), maxx);
            maxy = std::max(std::get<3>(box), maxy);
        }
    }
    float s = m.scale;
    return std::tuple{minx*s,miny*s,maxx*s,maxy*s};
}

void MObject::draw2() const {
    ofPushMatrix();
    ofScale(scale);
    //for (auto &path : of_paths) path.draw(x/scale,y/scale);
    ofPopMatrix();
}

std::vector<std::string> find_files(std::string root, const std::string &pattern, int depth)
{
    std::regex r{pattern};
    std::vector<std::string> files;
    for (auto it = fs::recursive_directory_iterator(root);it!=fs::recursive_directory_iterator();++it) {
        auto &e = *it;
        if (it.depth()>depth) {it.pop(); continue;}
        if(e.is_regular_file() && std::regex_match(e.path().filename().string(), r)) {
            files.push_back(e.path().string());
        }
    }
    return files;
}

/*
void MObject::draw() const
{
    ofPushMatrix();
    ofTranslate(x,y);
    ofScale(scale);
    
    for (size_t i=0;i<paths.size();i++) {
        if (i>0) {
            auto &p1 = paths[i-1].back();
            auto &p2 = paths[i][0];
            ofSetColor(0,0,0,0);
            ofDrawLine(p1.x,p1.y,p2.x,p2.y);
        }
        if (filled[i]) ofFill(); else ofNoFill();
        if (stroke_width[i] == 0) {
            ofSetColor(fill_colors[i]);
            ofBeginShape();
            for (auto &p : paths[i]) ofVertex(p.x, p.y);
            ofEndShape();
        } else {
            ofSetLineWidth(stroke_width[i]);
            ofSetColor(stroke_colors[i]);
            auto p1 = paths[i][0];
            for (int j=1;j<paths[i].size();j++) {
                auto &p = paths[i][j];
                ofDrawLine(p1.x, p1.y, p.x, p.y);
                p1 = p;
            }
        }
    }
    ofPopMatrix();
}
*/

MObject::MObject(const std::string& filename)
{
    scale = 1;
    svg.load(filename);
    float minx = std::numeric_limits<float>::max();
    float miny = std::numeric_limits<float>::max();
    float maxx =std::numeric_limits<float>::min();
    float maxy = std::numeric_limits<float>::min();
    for (auto path : svg.getPaths()) {
        auto is_filled = path.isFilled();
        auto fill_color = path.getFillColor();
        auto stroke_color = path.getStrokeColor();
        auto stroke_width = path.getStrokeWidth();
        path.setStrokeWidth(1);
        filled.push_back(is_filled);
        fill_colors.push_back(fill_color);
        stroke_colors.push_back(stroke_color);
        this->stroke_width.push_back(stroke_width);
        SubPath sub_path;
        for (auto outline : path.getOutline()) {
            std::vector<ofPoint> svg_points;
            auto v = outline.getResampledByCount(500).getVertices();
            if (v.size()<500) v.push_back(v.back());
            for (size_t i=0;i<v.size();i++) svg_points.push_back(v[i]);
            sub_path.push_back(std::move(svg_points));
        }
        paths.push_back(std::move(sub_path));
    }
    float surface_area = -1;
    for (int i=0;i<paths.size();i++) {
        auto &sub_paths = paths[i];
        for (int j=0;j<sub_paths.size();j++) {
            auto &outline = sub_paths[j];
            auto box = get_bounding_box(outline);
            minx = std::min(minx, std::get<0>(box));
            miny = std::min(miny, std::get<1>(box));
            maxx = std::max(maxx, std::get<2>(box));
            maxy = std::max(maxy, std::get<3>(box));
            auto s = (maxx-minx)*(maxy-miny);
            outer_path = (s>surface_area) ? static_cast<void>((surface_area=s)),i : outer_path;
        }
    }
    this->h=maxy-miny;
    this->w=maxx-minx;
    for (auto &path : paths)
        for (auto &spath : path)
            for (auto &point : spath)
                point += ofPoint{-minx,-miny};
    this->x = 0;
    this->y = 0;
    of_paths = generate_path(*this);
}

void MObject::set_color(ofColor c) {
    for(auto &fc : fill_colors) fc = c;
    of_paths = generate_path(*this);
}

using Frame = std::vector<MObject>;
std::vector<Frame> frames;

void draw_frame(int i)
{
    for (const auto &m : frames[i]) m.draw2();
}

void scale_frame(Frame& frame, double scale)
{
    for (auto &m : frame) {
        m.x = m.x * scale;
        m.y = m.y * scale;
        m.scale = scale * m.scale;
        m.of_paths = generate_path(m);
    }
}

MObject translate(const MObject& m, std::function<ofPoint(ofPoint)> f)
{
    MObject m2 = m;
    for (size_t i=0;i<m.paths.size();i++) {
        auto &path = m.paths[i];
        for (size_t j=0;j<path.size();j++) {
            auto &sub_path = path[j];
            for (size_t k=0;k<sub_path.size();k++)
                m2.paths[i][j][k] = f(sub_path[k]*m.scale+glm::vec2{m.x,m.y});
        }
    }
    
    m2.scale = 1;
    m2.x = 0;
    m2.y = 0;
    auto box = get_bounding_box(m2);
    m2.w = std::get<2>(box)-std::get<0>(box);
    m2.h = std::get<3>(box)-std::get<1>(box);
    m2.of_paths = generate_path(m2);
    return m2;
}

MObject interpolate2mobjects(
                             const MObject& m1,
                             const MObject& m2, double tran_step)
{
    MObject inter_m = m1;
    int num_paths = std::min(m1.paths.size(), m2.paths.size());
    
    for (size_t i=0;i<num_paths;i++) {
        int num_subpaths = std::min(m1.paths[i].size(), m2.paths[i].size());
        for (size_t j=0;j<num_subpaths;j++) {
            int num_points = std::min(m1.paths[i][j].size(), m2.paths[i][j].size());
            auto &sub_path = m1.paths[i][j];
            std::cout << "m1.sub_path.size = " << m1.paths[i][j].size() << ", ";
            std::cout << "m2.sub_path.size = " << m2.paths[i][j].size() << std::endl;
                for (size_t k=0;k<num_points;k++) {
                    auto p1 = m1.paths[i][j][k]*m1.scale+ofPoint(m1.x,m1.y);
                    auto p2 = m2.paths[i][j][k]*m2.scale+ofPoint(m2.x,m2.y);
                    inter_m.paths[i][j][k]=p1.getInterpolated(p2, tran_step);
                }
        }
    }
    inter_m.x = 0;
    inter_m.y = 0;
    inter_m.scale = 1;
    auto box = get_bounding_box(m2);
    inter_m.w = std::get<2>(box)-std::get<0>(box);
    inter_m.h = std::get<3>(box)-std::get<1>(box);
    inter_m.of_paths = generate_path(inter_m);
    return inter_m;
}


MObject interpolate2mobjects2(
                             const MObject& m1,
                             const MObject& m2, double tran_step)
{
    MObject inter_m = m1;
    int num_paths = std::min(m1.paths.size(), m2.paths.size());
    
    for (size_t i=0;i<num_paths;i++) {
        int num_subpaths = std::min(m1.paths[i].size(), m2.paths[i].size());
        for (size_t j=0;j<num_subpaths;j++) {
            auto point_index = findNearestPoints(m1.paths[i][j], m2.paths[i][j], m1.scale,m2.scale,m1.x,m1.y,m2.x,m2.y);
            int num_points = std::min(m1.paths[i][j].size(), m2.paths[i][j].size());
            auto &sub_path = m1.paths[i][j];
                for (size_t k=0;k<num_points;k++) {
                    auto p1 = m1.paths[i][j][k]*m1.scale+ofPoint(m1.x,m1.y);
                    auto p2 = m2.paths[i][j][point_index[k]]*m2.scale+ofPoint(m2.x,m2.y);
                    inter_m.paths[i][j][k]=p1.getInterpolated(p2, tran_step);
                }
        }
    }
    inter_m.x = 0;
    inter_m.y = 0;
    inter_m.scale = 1;
    auto box = get_bounding_box(m2);
    inter_m.w = std::get<2>(box)-std::get<0>(box);
    inter_m.h = std::get<3>(box)-std::get<1>(box);
    inter_m.of_paths = generate_path(inter_m);
    return inter_m;
}

std::vector<Frame> interpolate2frames(
                                      const std::vector<MObject>& f1,
                                      const std::vector<MObject>& f2,
                                      double tran_step)
{
    std::vector<Frame> frames;
    for (double k=0;k<1.0f;k+=tran_step) {
        Frame frame;
        for (size_t i=0;i<f1.size();i++) {
            auto m = interpolate2mobjects(f1[i], f2[i], k);
            frame.push_back(m);
        }
        frames.push_back(std::move(frame));
    }
    return frames;
}

std::tuple<float,float,float,float> get_frame_bounding_box(const Frame& frame) {
    float minx = std::numeric_limits<float>::max();
    float maxx = std::numeric_limits<float>::min();
    float miny = std::numeric_limits<float>::max();
    float maxy = std::numeric_limits<float>::min();
    for (auto &m : frame) {
        minx = std::min(minx,m.x);
        maxx = std::max(maxx, m.x+m.w*m.scale);
        miny = std::min(miny, m.y);
        maxy = std::max(maxy, m.y+m.h*m.scale);
    }
    return std::make_tuple(minx, miny, maxx, maxy);
}

void change_alpha(Frame& frame, int alpha)
{
    for (auto &m : frame) {
        for (auto &c : m.fill_colors) c.a = alpha;
        for (auto &c : m.stroke_colors) c.a = alpha;
        m.of_paths = generate_path(m);
    }
}

std::vector<int> findNearestPoints(const std::vector<ofPoint>& vectorA, const std::vector<ofPoint>& vectorB,float s1, float s2, float x1, float y1, float x2, float y2) {
    std::vector<int> result(vectorA.size(),-1);
    std::vector<int> taken(vectorA.size());
    for (int i=0;i<vectorA.size();i++) {
        int nearest = -1;
        auto a = vectorA[i]*s1 + ofPoint(x1,y1);
        ofPoint np;
        for (int j=0;j<vectorB.size();j++) {
            if (taken[j]) continue;
            auto b = vectorB[j]*s2+ofPoint(x2,y2);
            if (nearest>=0) np = vectorB[nearest]*s2+ofPoint(x2,y2);
            if (nearest==-1 || a.distance(np)>a.distance(b)) {
                
                nearest = j;
            }
        }
        result[i] = nearest;
        taken[nearest]=1;
    }
    return result;
}


void PiCreature::move_pupil(<#double x#>, <#double y#>)
{
    auto left = get_bounding_box(paths[left_eye]);
    auto right = get_bounding_box(paths[right_eye]);
}
