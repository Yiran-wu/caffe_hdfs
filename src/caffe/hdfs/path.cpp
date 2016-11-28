/*
 * path.cc
 * Copyright (C) 2016 qingze <qingze@node39.com>
 *
 * Distributed under terms of the MIT license.
 */

#include "caffe/hdfs/path.h"
#include "caffe/hdfs/scanner.h"

namespace caffe {

    void ParseURI(StringPiece remaining, StringPiece* scheme, StringPiece* host,
            StringPiece* path) {
        if (!Scanner(remaining).One(Scanner::LETTER)
                .Many(Scanner::LETTER_DIGIT_DOT)
                .StopCapture()
                .OneLiteral("://")
                .GetResult(&remaining, scheme)) {
            // If there's no scheme, assume the entire string is a path.
            *scheme = StringPiece(remaining.begin(), 0);
            *host = StringPiece(remaining.begin(), 0);
            *path = remaining;
            return;
        }

        // 1. Parse host
        if (!Scanner(remaining).ScanUntil('/').GetResult(&remaining, host)) {
            // No path, so the rest of the URI is the host.
            *host = remaining;
            *path = StringPiece(remaining.end(), 0);
            return;
        }

        // 2. The rest is the path
        *path = remaining;
    }


    std::pair<StringPiece, StringPiece> SplitPath(StringPiece uri) {
        StringPiece scheme, host, path;
        ParseURI(uri, &scheme, &host, &path);

        auto pos = path.rfind('/');
#ifdef PLATFORM_WINDOWS
        if (pos == StringPiece::npos)
            pos = path.rfind('\\');
#endif
        // Handle the case with no '/' in 'path'.
        if (pos == StringPiece::npos)
            return std::make_pair(StringPiece(uri.begin(), host.end() - uri.begin()),
                    path);

        // Handle the case with a single leading '/' in 'path'.
        if (pos == 0)
            return std::make_pair(
                    StringPiece(uri.begin(), path.begin() + 1 - uri.begin()),
                    StringPiece(path.data() + 1, path.size() - 1));

        return std::make_pair(
                StringPiece(uri.begin(), path.begin() + pos - uri.begin()),
                StringPiece(path.data() + pos + 1, path.size() - (pos + 1)));
    }

    StringPiece Basename(StringPiece path) {
        return SplitPath(path).second;
    }

}
