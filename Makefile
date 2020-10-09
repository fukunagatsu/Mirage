CXXFLAGS = -O3

Mirage: main.cpp mirage_parameters.cpp mirage.cpp

	$(CXX) $(CXXFLAGS) -o Mirage main.cpp mirage_parameters.cpp mirage.cpp -std=c++11  -I ./Eigen