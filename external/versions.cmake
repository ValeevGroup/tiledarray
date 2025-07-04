# for each dependency track both current and previous id (the variable for the latter must contain PREVIOUS)
# to be able to auto-update them

set(TA_TRACKED_VGCMAKEKIT_TAG 4c949fd7ccfe4b4f0e103288a5c0f557c6e740c0)

# N.B. may need to update INSTALL.md manually with the CUDA-specific version
set(TA_TRACKED_EIGEN_VERSION 3.3.5)
set(TA_TRACKED_EIGEN_PREVIOUS_VERSION 3.3)
set(TA_INSTALL_EIGEN_VERSION 3.4.0)
set(TA_INSTALL_EIGEN_PREVIOUS_VERSION 3.3.7)
set(TA_INSTALL_EIGEN_URL_HASH SHA256=b4c198460eba6f28d34894e3a5710998818515104d6e74e5cc331ce31e46e626)
set(TA_INSTALL_EIGEN_PREVIOUS_URL_HASH MD5=b9e98a200d2455f06db9c661c5610496)

set(TA_TRACKED_MADNESS_URL https://github.com/m-a-d-n-e-s-s/madness.git CACHE STRING "GIT_REPOSITORY for cloning MADNESS source")
set(TA_TRACKED_MADNESS_TAG 8abd78b8a304a88b951449d8cb127f5a91f27721 CACHE STRING "GIT_TAG (branch or hash) for cloning MADNESS")
set(TA_TRACKED_MADNESS_PREVIOUS_TAG bd84a52766ab497dedc2f15f2162fb0eb7ec4653)
set(TA_TRACKED_MADNESS_VERSION 0.10.1)
set(TA_TRACKED_MADNESS_PREVIOUS_VERSION 0.10.1)

set(TA_TRACKED_BTAS_TAG 62d57d9b1e0c733b4b547bc9cfdd07047159dbca)
set(TA_TRACKED_BTAS_PREVIOUS_TAG 1cfcb12647c768ccd83b098c64cda723e1275e49)

set(TA_TRACKED_LIBRETT_TAG 6eed30d4dd2a5aa58840fe895dcffd80be7fbece)
set(TA_TRACKED_LIBRETT_PREVIOUS_TAG 354e0ccee54aeb2f191c3ce2c617ebf437e49d83)

set(TA_TRACKED_UMPIRE-CXX-ALLOCATOR_TAG cbb08408b1cfbbacc24992e36f52edb3a29bdedc)
set(TA_TRACKED_UMPIRE-CXX-ALLOCATOR_PREVIOUS_TAG a48ad360e20b9733263768b54aa24afe5894faa4)

set(TA_TRACKED_SCALAPACKPP_TAG 6397f52cf11c0dfd82a79698ee198a2fce515d81)
set(TA_TRACKED_SCALAPACKPP_PREVIOUS_TAG 711ef363479a90c88788036f9c6c8adb70736cbf )

set(TA_TRACKED_RANGEV3_TAG 0.12.0)
set(TA_TRACKED_RANGEV3_PREVIOUS_TAG 2e0591c57fce2aca6073ad6e4fdc50d841827864)

set(TA_TRACKED_TTG_URL https://github.com/TESSEorg/ttg)
set(TA_TRACKED_TTG_TAG 3fe4a06dbf4b05091269488aab38223da1f8cb8e)
set(TA_TRACKED_TTG_PREVIOUS_TAG 26da9b40872660b864794658d4fdeee1a95cb4d6)

# oldest Boost we can tolerate ... old is fine but if Boost is missing build it requires something much younger
# SeQuant requires at least 1.81, so go with that
set(TA_OLDEST_BOOST_VERSION 1.81)
