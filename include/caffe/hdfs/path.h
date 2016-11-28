/*
 * path.h
 * Copyright (C) 2016 qingze <qingze@node39.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef PATH_H
#define PATH_H


#include "stringpiece.h"

namespace caffe {
    void ParseURI(StringPiece remaining, StringPiece* scheme, StringPiece* host, StringPiece* path);
    StringPiece Basename(StringPiece path);
}

#endif // PATH_H
