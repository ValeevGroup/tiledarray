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

def dots_to_undescores(str):
    return str.replace('.', '_')

def escape_special_chars(str):
    #str = str.replace('(', '\(')
    #str = str.replace(')', '\)')
    str = str.replace('/', '\/')
    str = str.replace('.', '\.')
    return str

def replace_dep_id(topsrc, file_ext, dep_name, old_id, new_id, search_prefix = '', search_postfix = ''):
    any_files_changed = False
    if old_id != new_id:
        # always exclude the versions file
        seek_retcode = os.system('grep -q -r --include="*.' + file_ext + '" --exclude="' + topsrc + '/external/versions.cmake" "' + search_prefix + old_id + search_postfix + '" ' + topsrc)
        if os.WIFEXITED(seek_retcode) and os.WEXITSTATUS(seek_retcode) == 0:
            any_files_changed = True
            print('changing ' + dep_name + ' id from', old_id, 'to', new_id)
            esc_search_prefix = escape_special_chars(search_prefix)
            esc_search_postfix = escape_special_chars(search_postfix)
            os.system('find ' + topsrc + ' -type f -name "*.' + file_ext + '" -print0 | xargs -0 sed -i \'\' -e \'s/' + esc_search_prefix + old_id + esc_search_postfix + '/' + esc_search_prefix + new_id + esc_search_postfix + '/g\'')
    return any_files_changed

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
        if len(tokens) < 3:
            continue
        if tokens[1].find('TRACKED_BOOST') != -1:
            if tokens[1].find('PREVIOUS') != -1:
                boost_old_version = tokens[2]
            else:
                boost_new_version = tokens[2]
        elif tokens[1].find('INSTALL_BOOST') != -1:
            if tokens[1].find('VERSION') != -1:
                if tokens[1].find('PREVIOUS') != -1:
                    boost_old_install_version = tokens[2]
                else:
                    boost_new_install_version = tokens[2]
            else:  # URL_HASH
                if tokens[1].find('PREVIOUS') != -1:
                    boost_old_install_url_hash = tokens[2]
                else:
                    boost_new_install_url_hash = tokens[2]
        elif tokens[1].find('TRACKED_EIGEN') != -1:
            if tokens[1].find('PREVIOUS') != -1:
                eigen_old_version = tokens[2]
            else:
                eigen_new_version = tokens[2]
        elif tokens[1].find('INSTALL_EIGEN') != -1:
            if tokens[1].find('VERSION') != -1:
                if tokens[1].find('PREVIOUS') != -1:
                    eigen_old_install_version = tokens[2]
                else:
                    eigen_new_install_version = tokens[2]
            else:  # URL_HASH
                if tokens[1].find('PREVIOUS') != -1:
                    eigen_old_install_url_hash = tokens[2]
                else:
                    eigen_new_install_url_hash = tokens[2]
        elif tokens[1].find('MADNESS') != -1 and tokens[1].find('_TAG') != -1:
            if tokens[1].find('PREVIOUS') != -1:
                madness_old_tag = tokens[2]
            else:
                madness_new_tag = tokens[2]
        elif tokens[1].find('MADNESS') != -1 and tokens[1].find('_VERSION') != -1:
            if tokens[1].find('PREVIOUS') != -1:
                madness_old_version = tokens[2]
            else:
                madness_new_version = tokens[2]
        elif tokens[1].find('BTAS') != -1:
            if tokens[1].find('PREVIOUS') != -1:
                btas_old_tag = tokens[2]
            else:
                btas_new_tag = tokens[2]
        elif tokens[1].find('LIBRETT') != -1:
            if tokens[1].find('PREVIOUS') != -1:
                librett_old_tag = tokens[2]
            else:
                librett_new_tag = tokens[2]
        elif tokens[1].find('UMPIRE') != -1:
            if tokens[1].find('PREVIOUS') != -1:
                umpire_old_tag = tokens[2]
            else:
                umpire_new_tag = tokens[2]
        elif tokens[1].find('BLACSPP') != -1:
            if tokens[1].find('PREVIOUS') != -1:
                blacspp_old_tag = tokens[2]
            else:
                blacspp_new_tag = tokens[2]
        elif tokens[1].find('SCALAPACKPP') != -1:
            if tokens[1].find('PREVIOUS') != -1:
                scalapackpp_old_tag = tokens[2]
            else:
                scalapackpp_new_tag = tokens[2]

any_files_changed = False

# Boost version in INSTALL.md
any_files_changed |= replace_dep_id(topsrc, 'md', 'Boost', boost_old_version, boost_new_version, 'boost.org/), version ', ' or higher')

# Eigen version in INSTALL.md
any_files_changed |= replace_dep_id(topsrc, 'md', 'Eigen', eigen_old_version, eigen_new_version, 'eigen.tuxfamily.org), version ', ' or higher')
# Eigen install version in eigen.cmake
any_files_changed |= replace_dep_id(topsrc, 'cmake', 'Eigen', eigen_old_install_version, eigen_new_install_version, 'libeigen/eigen/-/archive', '.tar.bz2')
any_files_changed |= replace_dep_id(topsrc, 'cmake', 'Eigen', eigen_old_install_url_hash, eigen_new_install_url_hash, 'MD5=', '')

# MADNESS version in tiledarray-config.cmake.in
any_files_changed |= replace_dep_id(topsrc, 'cmake.in', 'MADNESS', madness_old_version, madness_new_version, 'find_package(MADNESS ', ' ')
# MADNESS tag in INSTALL.md
any_files_changed |= replace_dep_id(topsrc, 'md', 'MADNESS', madness_old_tag, madness_new_tag, 'm-a-d-n-e-s-s/madness), tag ', ' ')

# BTAS tag in INSTALL.md
any_files_changed |= replace_dep_id(topsrc, 'md', 'BTAS', btas_old_tag, btas_new_tag, 'ValeevGroup/BTAS), tag ', '')

# libreTT tag in INSTALL.md
any_files_changed |= replace_dep_id(topsrc, 'md', 'libreTT', librett_old_tag, librett_new_tag, '', '')

# Umpire tag in INSTALL.md
any_files_changed |= replace_dep_id(topsrc, 'md', 'Umpire', umpire_old_tag, umpire_new_tag, '', '')

# BLACSPP tag in INSTALL.md
any_files_changed |= replace_dep_id(topsrc, 'md', 'BLACSPP', blacspp_old_tag, blacspp_new_tag, '', '')

# SCALAPACKPP tag in INSTALL.md
any_files_changed |= replace_dep_id(topsrc, 'md', 'SCALAPACKPP', scalapackpp_old_tag, scalapackpp_new_tag, '', '')

if any_files_changed:
    sys.exit(1)
else:
    sys.exit(0)
