/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "caffe/hdfs/error.h"

#include <errno.h>
#include <string.h>

namespace  caffe {

    Code ErrnoToCode(int err_number) {
        Code code;
        switch (err_number) {
            case 0:
                code = Code::OK;
                break;
            case EINVAL:        // Invalid argument
            case ENAMETOOLONG:  // Filename too long
            case E2BIG:         // Argument list too long
            case EDESTADDRREQ:  // Destination address required
            case EDOM:          // Mathematics argument out of domain of function
            case EFAULT:        // Bad address
            case EILSEQ:        // Illegal byte sequence
            case ENOPROTOOPT:   // Protocol not available
            case ENOSTR:        // Not a STREAM
            case ENOTSOCK:      // Not a socket
            case ENOTTY:        // Inappropriate I/O control operation
            case EPROTOTYPE:    // Protocol wrong type for socket
            case ESPIPE:        // Invalid seek
                code = Code::INVALID_ARGUMENT;
                break;
            case ETIMEDOUT:  // Connection timed out
            case ETIME:      // Timer expired
                code = Code::DEADLINE_EXCEEDED;
                break;
            case ENODEV:  // No such device
            case ENOENT:  // No such file or directory
            case ENXIO:   // No such device or address
            case ESRCH:   // No such process
                code = Code::NOT_FOUND;
                break;
            case EEXIST:         // File exists
            case EADDRNOTAVAIL:  // Address not available
            case EALREADY:       // Connection already in progress
                code = Code::ALREADY_EXISTS;
                break;
            case EPERM:   // Operation not permitted
            case EACCES:  // Permission denied
            case EROFS:   // Read only file system
                code = Code::PERMISSION_DENIED;
                break;
            case ENOTEMPTY:   // Directory not empty
            case EISDIR:      // Is a directory
            case ENOTDIR:     // Not a directory
            case EADDRINUSE:  // Address already in use
            case EBADF:       // Invalid file descriptor
            case EBUSY:       // Device or resource busy
            case ECHILD:      // No child processes
            case EISCONN:     // Socket is connected
#if !defined(_WIN32)
            case ENOTBLK:     // Block device required
#endif
            case ENOTCONN:    // The socket is not connected
            case EPIPE:       // Broken pipe
#if !defined(_WIN32)
            case ESHUTDOWN:   // Cannot send after transport endpoint shutdown
#endif
            case ETXTBSY:     // Text file busy
                code = Code::FAILED_PRECONDITION;
                break;
            case ENOSPC:   // No space left on device
#if !defined(_WIN32)
            case EDQUOT:   // Disk quota exceeded
#endif
            case EMFILE:   // Too many open files
            case EMLINK:   // Too many links
            case ENFILE:   // Too many open files in system
            case ENOBUFS:  // No buffer space available
            case ENODATA:  // No message is available on the STREAM read queue
            case ENOMEM:   // Not enough space
            case ENOSR:    // No STREAM resources
#if !defined(_WIN32)
            case EUSERS:   // Too many users
#endif
                code = Code::RESOURCE_EXHAUSTED;
                break;
            case EFBIG:      // File too large
            case EOVERFLOW:  // Value too large to be stored in data type
            case ERANGE:     // Result too large
                code = Code::OUT_OF_RANGE;
                break;
            case ENOSYS:           // Function not implemented
            case ENOTSUP:          // Operation not supported
            case EAFNOSUPPORT:     // Address family not supported
#if !defined(_WIN32)
            case EPFNOSUPPORT:     // Protocol family not supported
#endif
            case EPROTONOSUPPORT:  // Protocol not supported
#if !defined(_WIN32)
            case ESOCKTNOSUPPORT:  // Socket type not supported
#endif
            case EXDEV:            // Improper link
                code = Code::UNIMPLEMENTED;
                break;
            case EAGAIN:        // Resource temporarily unavailable
            case ECONNREFUSED:  // Connection refused
            case ECONNABORTED:  // Connection aborted
            case ECONNRESET:    // Connection reset
            case EINTR:         // Interrupted function call
#if !defined(_WIN32)
            case EHOSTDOWN:     // Host is down
#endif
            case EHOSTUNREACH:  // Host is unreachable
            case ENETDOWN:      // Network is down
            case ENETRESET:     // Connection aborted by network
            case ENETUNREACH:   // Network unreachable
            case ENOLCK:        // No locks available
            case ENOLINK:       // Link has been severed
#if !(defined(__APPLE__) || defined(_WIN32))
            case ENONET:  // Machine is not on the network
#endif
                code = Code::UNAVAILABLE;
                break;
            case EDEADLK:  // Resource deadlock avoided
#if !defined(_WIN32)
            case ESTALE:   // Stale file handle
#endif
                code = Code::ABORTED;
                break;
            case ECANCELED:  // Operation cancelled
                code = Code::CANCELLED;
                break;
                // NOTE: If you get any of the following (especially in a
                // reproducible way) and can propose a better mapping,
                // please email the owners about updating this mapping.
            case EBADMSG:      // Bad message
            case EIDRM:        // Identifier removed
            case EINPROGRESS:  // Operation in progress
            case EIO:          // I/O error
            case ELOOP:        // Too many levels of symbolic links
            case ENOEXEC:      // Exec format error
            case ENOMSG:       // No message of the desired type
            case EPROTO:       // Protocol error
#if !defined(_WIN32)
            case EREMOTE:      // Object is remote
#endif
                code = Code::UNKNOWN;
                break;
            default: {
                         code = Code::UNKNOWN;
                         break;
                     }
        }
        return code;
    }

    Status IOError(const std::string& context, int err_number) {
        auto code = ErrnoToCode(err_number);
        if (code == Code::UNKNOWN) {
            return Status(code, context + "; " + strerror(err_number));
        } else {
            return Status(code, context);
        }
    }

}  // namespace
