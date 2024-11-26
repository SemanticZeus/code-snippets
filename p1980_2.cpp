#include <gmpxx.h>
#include <iostream>
#include <cmath> // For floor

int main() {
    // Set the precision for the calculations
    mpf_set_default_prec(2048); // 2048 bits of precision

    // Define the base components
    mpf_class sqrt2, sqrt3, base, result;
    mpf_sqrt(sqrt2.get_mpf_t(), mpf_class(2).get_mpf_t()); // sqrt(2)
    mpf_sqrt(sqrt3.get_mpf_t(), mpf_class(3).get_mpf_t()); // sqrt(3)
    base = sqrt2 + sqrt3; // sqrt(2) + sqrt(3)
    std::cout << "base = " << base.get_ui() << std::endl;
    // Raise the base to the power of 1980
    mpf_pow_ui(result.get_mpf_t(), base.get_mpf_t(), 1980);

    // Multiply by 10
    result *= 10;

    // Get the modulus 100
    mpf_class modulus = 100;
    mpf_class remainder = result - modulus * mpf_class(floor(result / modulus));

    // Convert remainder to an integer to display the final result
    unsigned long final_result = remainder.get_ui();

    // Output the result
    std::cout << "Final Result: " << final_result << std::endl;

    return 0;
}

