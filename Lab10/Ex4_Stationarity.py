import numpy

def get_roots(coefficients):
    coeffs = numpy.array(coefficients, dtype=float)
    coeffs = coeffs / coeffs[-1]
    N = len(coeffs) - 1
    if N == 0:
        return numpy.array([])
    
    companion = numpy.zeros((N, N))
    companion[1:, :-1] = numpy.eye(N-1)
    companion[:, -1] = -coeffs[:-1]
    # print(companion)
    return numpy.linalg.eigvals(companion)

def is_stationary(coefficients):
    char_poly = numpy.concatenate([[1], -numpy.array(coefficients)])
    roots = get_roots(char_poly)
    return numpy.all(numpy.abs(roots) > 1)

# x = [1, 2, 1]
# print(get_roots(x))
# print(is_stationary(x))