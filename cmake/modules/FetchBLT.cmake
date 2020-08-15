FetchContent_Declare(
    BLT
    GIT_REPOSITORY      https://github.com/evaleev/blt.git
    GIT_TAG             origin/develop
)
FetchContent_MakeAvailable(BLT)
FetchContent_GetProperties(BLT
    SOURCE_DIR BLT_SOURCE_DIR
)
