#ifndef CUBIN_CPU_H
#define CUBIN_CPU_H

#include <vector>
#include <iostream>

template<typename bit_vector_t>
int computeDistanceCPU(const vector<bit_vector_t> &Ab,
                        const vector<bit_vector_t> &Bb,
                        const vector<bit_vector_t> &Cb,
                        const int height,
                        const int width)
{
    int error = 0;

    #pragma omp parallel for reduction(+:error)
    for(int j=0; j < width; ++j) {
        uint32_t B_j = Bb[j];
        for(int i=0; i < height; ++i) {
            const int product = (Ab[i] & B_j) ? 1 : 0;

            const int vecId = i / 32 * width + j;
            const int vecLane = i % 32;
            const int C_ij = (Cb[vecId] >> vecLane) & 1;

            error += product ^ C_ij;
        }
    }

    return error;
}

template<typename bit_vector_t>
int vectorMatrixMultCompareRowCPU(vector<bit_vector_t> &Ab,
                                  const vector<bit_vector_t> &Bb,
                                  const vector<bit_vector_t> &Cb,
                                  const int height,
                                  const int width,
                                  const uint8_t factorDim,
                                  const int startrow,
                                  const int numrows,
                                  const uint32_t seed, 
                                  const float temperature,
                                  const float flipManyChance,
                                  const uint32_t flipManyDepth,
                                  const int inverse_density)
{
    int error_update = 0;

    #pragma omp for
    // #pragma omp parallel for reduction(+:error_update)
    for(int id=0; id < numrows; ++id) {
        const int i = (startrow + id) % height;

        fast_kiss_state32_t state;
        state = get_initial_fast_kiss_state32(seed + id);

        bit_vector_t A_i = Ab[i];
        bit_vector_t A_i_changed = Ab[i] ^ get_flip_mask(factorDim, state, flipManyChance, flipManyDepth);

        int error = 0;
        for(int j=0; j < width; ++j) {
            const int product_new = (A_i_changed & Bb[j]) ? 1 : 0;
            const int product_old = (A_i         & Bb[j]) ? 1 : 0;

            const int vecId = i / 32 * width + j;
            const int vecLane = i % 32;
            const int C_ij = (Cb[vecId] >> vecLane) & 1;

            error += error_measure(product_new, C_ij, inverse_density)
                   - error_measure(product_old, C_ij, inverse_density);
        }

        if (metro(state, error, temperature, width)) {
            Ab[i] = A_i_changed;
            error_update += error;
        }
    }

    return error_update;
}

template<typename bit_vector_t>
int vectorMatrixMultCompareColCPU(const vector<bit_vector_t> &Ab,
                                  vector<bit_vector_t> &Bb,
                                  const vector<bit_vector_t> &Cb,
                                  const int height,
                                  const int width,
                                  const uint8_t factorDim,
                                  const int startcol,
                                  const int numcols,
                                  const uint32_t seed, 
                                  const float temperature,
                                  const float flipManyChance,
                                  const uint32_t flipManyDepth,
                                  const int inverse_density)
{
    int error_update = 0;

    #pragma omp for
    // #pragma omp parallel for reduction(+:error_update)
    for(int id=0; id < numcols; ++id) {
        const int j = (startcol + id) % width;

        fast_kiss_state32_t state;
        state = get_initial_fast_kiss_state32(seed + id);

        bit_vector_t B_j = Bb[j];
        bit_vector_t B_j_changed = Bb[j] ^ get_flip_mask(factorDim, state, flipManyChance, flipManyDepth);

        int error = 0;
        for(int i=0; i < height; ++i) {
            const int product_new = (Ab[i] & B_j_changed) ? 1 : 0;
            const int product_old = (Ab[i] & B_j        ) ? 1 : 0;

            const int vecId = i / 32 * width + j;
            const int vecLane = i % 32;
            const int C_ij = (Cb[vecId] >> vecLane) & 1;

            error += error_measure(product_new, C_ij, inverse_density)
                   - error_measure(product_old, C_ij, inverse_density);
        }

        if (metro(state, error, temperature, height)) {
            Bb[j] = B_j_changed;
            error_update += error;
        }
    }

    return error_update;
}

struct coo {
    coo(uint32_t x, uint32_t y) : x_{x}, y_{y} {}

    uint32_t x_;
    uint32_t y_;
};

vector<coo> computeProductCOO(const vector<uint32_t> &Ab,
                              const vector<uint32_t> &Bb,
                              const int height,
                              const int width)
{
    vector<coo> C;

    #pragma omp parallel for ordered schedule(static,1)
    for(int i=0; i < height; ++i) {
        uint32_t row = Ab[i];
        vector<coo> Ci;
        for(int j=0; j < width; ++j) {
            if(row & Bb[j])
                Ci.emplace_back(i,j);
        }
        #pragma omp ordered
        C.insert(C.end(), Ci.begin(), Ci.end());
    }
    return C;
}


template<typename factor_t = uint32_t>
class Cubin_CPU
{
    using factor_matrix_t = std::vector<factor_t>;
    using bit_vector_t = uint32_t;
    using bit_matrix_t = std::vector<bit_vector_t>;

public:
    Cubin_CPU(const factor_matrix_t& A,
          const factor_matrix_t& B,
          const bit_matrix_t& C,
          const uint8_t factorDim = 20,
          const float density = 0.99f)
    {
        cout << "~~~ CPU CuBin ~~~" << endl; 

        if(factorDim > 32) {
            std::cerr << "Factor dimension too big! Maximum is 32." << endl;
            factorDim_ = 32;
        }
        else factorDim_ = factorDim;

        inverse_density_ = 1 / density;

        initialize(A, B, C);
    }

    ~Cubin_CPU() {
        clear();
    }

    bool initialize(const factor_matrix_t& A, const factor_matrix_t& B, const bit_matrix_t& C) {
        if (std::is_same<factor_t, uint32_t>::value) {
            lineSize_ = 1;
            lineSize_padded_ = 1;
        }
        if(std::is_same<factor_t, float>::value) {
            lineSize_ = factorDim_;
            lineSize_padded_ = factorDim_;
        }

        if( SDIV(A.size()/lineSize_,32) * B.size()/lineSize_ != C.size()) {
            std::cerr << "CuBin construction: Matrix dimension mismatch." << std::endl;
            return false;
        }

        if(initialized_) {
            std::cerr << "CuBin already initialized. Please clear CuBin before reinitialization." << std::endl;
            return false;
        }

        height_ = A.size() / lineSize_;

        width_ = B.size() / lineSize_;
        
        d_A = A;
        d_B = B;
        d_C = C;

        distance_ = computeDistanceCPU(d_A, d_B, d_C, height_, width_);

        std::cout << "CuBin initialization complete." << endl;

        std::cout << "Matrix dimensions:\t" << height_ << "x" << width_ << endl;
        std::cout << "Factor dimension:\t" << (int) factorDim_ << endl;

        std::cout << "Start distance: "
                  << "\tabs_err: " << distance_
                  << "\trel_err: " << (float) distance_ / (height_ * width_)
                  << std::endl;

        return initialized_ = true;
    }

    bool verifyDistance() {
        if(!initialized_) {
            std::cerr << "CuBin not initialized." << endl;
            return false;
        }

        int distance_proof;

        distance_proof = computeDistanceCPU(d_A, d_B, d_C, height_, width_);

        bool equal = distance_ == distance_proof;
        if(!equal) {
            std::cout << "----- !Distances differ! -----\n";
            std::cout << "Running distance:  " << distance_ << "\n";
            std::cout << "Real distance:     " << distance_proof << std::endl;
        }
        return equal;
    } 

    void clear() {
        if(initialized_) {
            d_A.clear();
            d_B.clear();
            d_C.clear();
            distance_ = 0;
            initialized_ = false;
        }
    }

    void getFactors(factor_matrix_t& A, factor_matrix_t& B) {
        if(!initialized_) {
            std::cerr << "CuBin not initialized." << endl;
            return;
        }

        A = d_A;
        B = d_B;
    }

    int getDistance() {
        if(!initialized_) {
            std::cerr << "CuBin not initialized." << endl;
            return -1;
        }
        return distance_;
    }

    struct CuBin_config {
        size_t verbosity = 1;
        size_t linesAtOnce = 0;
        size_t maxIterations = 0;
        int distanceThreshold = 0;
        size_t distanceShowEvery = std::numeric_limits<size_t>::max();
        float tempStart = 0.0f;
        float tempEnd = -1.0f;
        float tempFactor = 0.98f;
        size_t tempStep = std::numeric_limits<size_t>::max();
        uint32_t seed = 0;
        bool loadBalance = false;
        float flipManyChance = 0.1f;
        uint32_t flipManyDepth = 2;
        size_t stuckIterationsBeforeBreak = std::numeric_limits<size_t>::max();
    };

    void run(const CuBin_config& config) {
        if(!initialized_) {
            std::cerr << "CuBin not initialized." << endl;
            return;
        }

        size_t linesAtOnce = config.linesAtOnce;

        if(config.verbosity > 0) {
            std::cout << "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n";
            std::cout << "- - - - Starting " << config.maxIterations
                      << " CPU iterations, changing " << linesAtOnce
                      << " lines each time\n";
            std::cout << "- - - - Showing error every " << config.distanceShowEvery
                      << " steps\n";
            if(config.tempStart > 0) {
                std::cout << "- - - - Start temperature " << config.tempStart
                          << " multiplied by " << config.tempFactor
                          << " every " << config.tempStep
                          << " steps\n";

            }
            std::cout << "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -";
            std::cout << std::endl;
        }

        fast_kiss_state32_t state = get_initial_fast_kiss_state32(config.seed);
        float temperature = config.tempStart;
        size_t iteration = 0;
        size_t stuckIterations = 0;
        auto distancePrev = distance_;
        int lineToBeChanged;
        uint32_t cpuSeed;
        #pragma omp parallel private(iteration)
        while( distance_ > config.distanceThreshold
                && iteration++ < config.maxIterations
                && temperature > config.tempEnd
                && stuckIterations < config.stuckIterationsBeforeBreak)
        {
            // Change rows
            #pragma omp single
            {
                lineToBeChanged = (fast_kiss32(state) % height_) / WARPSPERBLOCK * WARPSPERBLOCK;
                cpuSeed = fast_kiss32(state) + iteration;
            }

            int distance_update = vectorMatrixMultCompareRowCPU(d_A, d_B, d_C, height_, width_, factorDim_,
                                          lineToBeChanged, min(linesAtOnce, height_), cpuSeed, temperature/10,
                                          config.flipManyChance, config.flipManyDepth, inverse_density_);

            // Change cols
            #pragma omp single
            {
                lineToBeChanged = (fast_kiss32(state) % width_) / WARPSPERBLOCK * WARPSPERBLOCK;
                cpuSeed = fast_kiss32(state) + iteration;
            }

            distance_update += vectorMatrixMultCompareColCPU(d_A, d_B, d_C, height_, width_, factorDim_,
                                          lineToBeChanged, min(linesAtOnce, height_), cpuSeed, temperature/10,
                                          config.flipManyChance, config.flipManyDepth, inverse_density_);

            #pragma omp atomic
            distance_ += distance_update;

            #pragma omp single
            {
                if(config.verbosity > 0 && iteration % config.distanceShowEvery == 0) {
                    std::cout << "Iteration: " << iteration
                              << "\tabs_err: " << distance_
                              << "\trel_err: " << (float) distance_ / (height_*width_)
                              << "\ttemp: " << temperature;
                    std::cout << std::endl;
                }

                if(iteration % config.tempStep == 0) {
                    temperature *= config.tempFactor;
                }
                if(distance_ == distancePrev)
                    stuckIterations++;
                else
                    stuckIterations = 0;
                distancePrev = distance_;
            }
        }

        if(config.verbosity > 0) {
            if (!(iteration < config.maxIterations))
                std::cout << "Reached iteration limit: " << config.maxIterations << std::endl;
            if (!(distance_ > config.distanceThreshold))
                std::cout << "Distance below threshold." << std::endl;
            if (!(temperature > config.tempEnd))
                std::cout << "Temperature below threshold." << std::endl;
            if (!(stuckIterations < config.stuckIterationsBeforeBreak))
                std::cout << "Stuck for " << stuckIterations << " iterations." << std::endl;
        }
        std::cout << "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n";
        std::cout << "Final result: "
                  << "\tabs_err: " << distance_
                  << "\trel_err: " << (float) distance_ / (height_ * width_)
                  << std::endl;
    }  

private:
    bool initialized_ = false;
    // factor_matrix_t A;
    // factor_matrix_t B;
    // bit_matrix_t C;
    factor_matrix_t d_A;
    factor_matrix_t d_B;
    bit_matrix_t d_C;
    int inverse_density_;
    int distance_;
    // size_t height_padded;
    uint8_t factorDim_ = 20;
    size_t height_ = 0;
    size_t width_ = 0;
    size_t lineSize_ = 1;
    size_t lineSize_padded_ = 1;
};

#endif
