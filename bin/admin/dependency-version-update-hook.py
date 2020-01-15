#!/usr/bin/env python

from __future__ import print_function
import sys
import os
import os.path as op

def to_base_version(full_version):
    base_version = full_version
    plus_pos = base_version.find('+')
    if plus_pos != -1:
        base_version = base_version[0:plus_pos]
    minus_pos = base_version.find('-')
    if minus_pos != -1:
        base_version = base_version[0:minus_pos]
    return base_version

argv = sys.argv
topsrc = op.normpath(op.join(op.abspath(op.dirname(sys.argv[0])), '../..'))
if len(argv) == 1:
    version_cmake_path = topsrc + '/external/versions.cmake'
elif len(argv) == 2:
    # no-op if given
    version_cmake_path = op.abspath(sys.argv[1])
    if op.basename(version_cmake_path) != 'versions.cmake':
        sys.exit(0)
else:
    print('invalid number of arguments')
    sys.exit(0)

# extract dependencies tags and versions
with open(version_cmake_path) as inf:
    for line in inf:
        line = line.replace('(', ' ')
        line = line.replace(')', ' ')
        tokens = line.split()
        if tokens[1].find('MADNESS') != -1 and tokens[1].find('_TAG') != -1:
            if tokens[1].find('PREVIOUS') != -1:
                madness_old_tag = tokens[2]
            else:
                madness_new_tag = tokens[2]
        elif tokens[1].find('MADNESS') != -1 and tokens[1].find('_VERSION') != -1:
            if tokens[1].find('PREVIOUS') != -1:
                madness_old_version = tokens[2]
            else:
                madness_new_version = tokens[2]
        elif tokens[1].find('BOOST') != -1 and tokens[1].find('_VERSION') != -1:
            if tokens[1].find('PREVIOUS') != -1:
                boost_old_version = tokens[2]
            else:
                boost_new_version = tokens[2]

any_files_changed = False

# replace Libint full version in INSTALL.md and *.sh scripts
# seek_retcode1 = os.system('grep -q -r --include="*.md" "' + libint_old_version + '" ' + topsrc)
# seek_retcode2 = os.system('grep -q -r --include="*.sh" "' + libint_old_version + '" ' + topsrc)
# if (os.WIFEXITED(seek_retcode1) and os.WEXITSTATUS(seek_retcode1) == 0) or (os.WIFEXITED(seek_retcode2) and os.WEXITSTATUS(seek_retcode2) == 0):
#     any_files_changed = True
#     print('changing Libint version from', libint_old_version, 'to', libint_new_version)
#     os.system('find ' + topsrc + ' -type f -name "*.md" -o -name "*.sh" -print0 | xargs -0 sed -i \'\' -e \'s/' + libint_old_version + '/' + libint_new_version + '/g\'')

# may need to replace Libint base version (without prerelease and build tags)
# libint_old_base_version = to_base_version(libint_old_version)
# libint_new_base_version = to_base_version(libint_new_version)
# if libint_old_base_version != libint_new_base_version:
#     if libint_old_base_version[0] != '2' or libint_new_base_version[0] != '2':
#         print('Major version change of Libint detected ... sorry, have to do this manually')
#         sys.exit(1)
#     seek_retcode = os.system('grep -q -r --include="*.cmake*" "Libint2 ' + libint_old_base_version + '" ' + topsrc)
#     if os.WIFEXITED(seek_retcode) and os.WEXITSTATUS(seek_retcode) == 0:
#         any_files_changed = True
#         print('changing Libint base version from', libint_old_base_version, 'to', libint_new_base_version)
#         os.system('find ' + topsrc + ' -type f -name "*.cmake*" -print0 | xargs -0 sed -i \'\' -e \'s/Libint2 ' + libint_old_base_version + '/Libint2 ' + libint_new_base_version + '/g\'')

if any_files_changed:
    sys.exit(1)
else:
    sys.exit(0)
