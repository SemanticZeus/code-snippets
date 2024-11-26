/* IMO LongList 1980
 * find the digits left and right of the decimal point in the decimal form of the number
 * (sqrt(3)+sqrt(2))**1980
 * */
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/multiprecision/cpp_int.hpp>

using namespace boost::multiprecision;
using hpf = boost::multiprecision::number<boost::multiprecision::cpp_bin_float<1000>>;

int main(int argc, char *argv[])
{
  boost::multiprecision::mpf_float a = 2;
  boost::multiprecision::mpf_float b = 3;

  boost::multiprecision::mpf_float c = sqrt(a)+sqrt(b);
  c = boost::multiprecision::pow(c,1980);
  c = c-100*boost::multiprecision::floor(c/10);
  //cpp_int d = static_cast<cpp_int>(c);
  std::cout << c  << std::endl;

  return 0;
}
