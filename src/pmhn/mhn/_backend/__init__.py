"""This subpackage is a modified version of the code
(available on the MIT License terms, see below) from the

https://github.com/cbg-ethz/metMHN 

package. We are grateful for the authors for creating their package. 


MIT License

Copyright (c) 2023 Computational Biology Group (CBG)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

from pmhn.mhn._backend.likelihood import (
    grad_loglikelihood_nonzero,
    grad_loglikelihood_zero,
    loglikelihood_nonzero,
    loglikelihood_zero,
)

__all__ = [
    "loglikelihood_nonzero",
    "loglikelihood_zero",
    "grad_loglikelihood_nonzero",
    "grad_loglikelihood_zero",
]
