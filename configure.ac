AC_INIT([cg], [2022-a05-ref], [selgrad])

AC_CONFIG_SRCDIR([src/main.cpp])
AC_CONFIG_HEADERS(config.h)
AC_CONFIG_AUX_DIR([auxx])

AM_INIT_AUTOMAKE(dist-bzip2 no-dist-gzip)

AC_PROG_CXX
AC_LANG([C++])
AX_CXX_COMPILE_STDCXX(11)

AC_CHECK_LIB([bigduckgl], [main])
AC_CHECK_LIB([imgui], [main])
AC_CHECK_LIB([GLEW], [main])
AC_CHECK_LIB([glfw], [main])
AC_CHECK_LIB([GL], [main])
AC_CHECK_LIB([assimp], [main])

AC_CONFIG_FILES([Makefile src/Makefile shaders/Makefile render-data/Makefile render-data/fonts/Makefile])

AC_OUTPUT
