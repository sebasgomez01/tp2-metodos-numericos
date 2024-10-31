#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <eigen3/Eigen/Dense>
#include <tuple>


using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
namespace py = pybind11;

tuple<double, VectorXd, int> powerIteration(MatrixXd A, int niter = 10000, double epsilon = 1e-6) {
    int n = A.rows();
    double eigenvalue;
    VectorXd v(n);
    v.setRandom();
    int pasos = 0;
    VectorXd v_anterior = v;
    for(int i = 0; i < niter; i++) {
        pasos++;
        v_anterior = v;
        v = A * v;
        v = v / v.norm();
        double norma_infinito = (v_anterior - v).lpNorm<Eigen::Infinity>();
        if(norma_infinito < epsilon) {
            cout << "La cantidad de iteraciones fue:" << i << endl;
            break;
            
        }

    }
    eigenvalue = (v.transpose()).dot(A*v);
    eigenvalue = eigenvalue / ((v.transpose()).dot(v));
    tuple<double, VectorXd, int> result(eigenvalue, v, pasos);
    return result;
}

pair<VectorXd, MatrixXd> eigen(MatrixXd A, int num = 2, int niter = 10000, double epsilon = 1e-6) {
    MatrixXd A_copy = A;
    VectorXd eigenvalues(num);
    MatrixXd eigenvectors(A.rows(), num);
    
    for(int i = 0; i < num; i++) {
        tuple<double, VectorXd, int> eigens = powerIteration(A_copy, niter, epsilon);
        eigenvalues(i) = get<0>(eigens);
        eigenvectors.col(i) = get<1>(eigens);
                    
        A_copy -= ((eigenvalues(i) * eigenvectors.col(i)) * eigenvectors.col(i).transpose());
    }
    pair<VectorXd, MatrixXd> result = make_pair(eigenvalues, eigenvectors);
    return result;
}

int main() {
    return 0;
}
// declaro el módulo de pybind
PYBIND11_MODULE(powerMethod, m) {
    m.def("eigen", &eigen, 
    py::arg("A"), py::arg("num") = 2, py::arg("niter") = 10000, py::arg("epsilon") = 1e-6,
    "Funcion que calcula los num autovalores y autovectores dominantes de A");

    m.def("powerIteration", &powerIteration, 
    py::arg("A"), py::arg("niter") = 10000, py::arg("epsilon") = 1e-6,
    "Funcion para calcular el autovalor y autovector dominantes con el metodo de la potencia");
}
