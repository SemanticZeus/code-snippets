/* IMO LongList 1980
 * find the digits left and right of the decimal point in the decimal form of the number
 * (sqrt(3)+sqrt(2))**1980
 * */
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/cpp_int.hpp>

using namespace boost::multiprecision;
using hpf = boost::multiprecision::number<boost::multiprecision::cpp_dec_float<10000>>;

int main(int argc, char *argv[])
{
  hpf a = 2;
  hpf b = 3;

  hpf c = sqrt(a)+sqrt(b);
  c = boost::multiprecision::pow(c,1980);
  c*=10;
  c = boost::multiprecision::floor(c);
  cpp_int d = static_cast<cpp_int>(c);
  std::cout << d%100  << std::endl;

  return 0;
}
