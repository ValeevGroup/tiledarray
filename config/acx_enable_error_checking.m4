AC_DEFUN([ACX_ENABLE_ERROR_CHECKING], [
  TA_DEFAULT_ERROR=3
  AC_ARG_ENABLE([error-checking],
    [AC_HELP_STRING([--enable-error-checking@<:@=throw|assert|no@:>@],
      [Enable default error checking@<:@default=throw@:>@.])],
    [
      case $enableval in
        yes)
          AC_DEFINE([TA_DEFAULT_ERROR], [1], 
            [Defines the default error checking behavior. none = 0, throw = 1, assert = 2])
        ;;
        throw)
          AC_DEFINE([TA_DEFAULT_ERROR], [1],
            [Defines the default error checking behavior. none = 0, throw = 1, assert = 2])
        ;;
        assert)
          AC_DEFINE([TA_DEFAULT_ERROR], [2],
            [Defines the default error checking behavior. none = 0, throw = 1, assert = 2])
        ;;
        no)
          AC_DEFINE([TA_DEFAULT_ERROR], [0],
            [Defines the default error checking behavior. none = 0, throw = 1, assert = 2])
        ;;
        *)
          AC_MSG_ERROR([Invalid input for error checking.])
        ;;
      esac   
    ],
    [AC_DEFINE([TA_DEFAULT_ERROR], [0],
      [Defines the default error checking behavior. none = 0, throw = 1, assert = 2])]
  )
])