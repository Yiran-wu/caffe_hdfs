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

#ifndef HADOOP_FILE_SYSTEM_H_
#define HADOOP_FILE_SYSTEM_H_

#include "hdfs.h"
#include "error.h"
#include "path.h"
#include "scanner.h"
#include "types.h"
#include "file_statistics.h"
#include <string>
#include <memory>
#include <dlfcn.h>

extern "C" {
    struct hdfs_internal;
    typedef hdfs_internal* hdfsFS;
}

namespace caffe {

    class LibHDFS {
        public:
            Status LoadLibrary(const char* library_filename, void** handle);
            Status GetSymbolFromLibrary(void* handle, const char* symbo_name, void** symbol);
            template <typename R, typename... Args>
            Status BindFunc(void* handle, const char* name, std::function<R(Args...)>* func);

            static LibHDFS* Load() {
                static LibHDFS* lib = []() -> LibHDFS* {
                    LibHDFS* lib = new LibHDFS;
                    lib->LoadAndBind();
                    return lib;
                }();

                return lib;
            }

            // The status, if any, from failure to load.
            Status status() { return status_; }

            std::function<hdfsFS(hdfsBuilder*)> hdfsBuilderConnect;
            std::function<hdfsBuilder*()> hdfsNewBuilder;
            std::function<void(hdfsBuilder*, const char*)> hdfsBuilderSetNameNode;
            std::function<void(hdfsBuilder*, const char* kerbTicketCachePath)>
                hdfsBuilderSetKerbTicketCachePath;
            std::function<int(hdfsFS, hdfsFile)> hdfsCloseFile;
            std::function<tSize(hdfsFS, hdfsFile, tOffset, void*, tSize)> hdfsPread;
            std::function<tSize(hdfsFS, hdfsFile, const void*, tSize)> hdfsWrite;
            std::function<int(hdfsFS, hdfsFile)> hdfsFlush;
            std::function<int(hdfsFS, hdfsFile)> hdfsHSync;
            std::function<hdfsFile(hdfsFS, const char*, int, int, short, tSize)>
                hdfsOpenFile;
            std::function<int(hdfsFS, const char*)> hdfsExists;
            std::function<hdfsFileInfo*(hdfsFS, const char*, int*)> hdfsListDirectory;
            std::function<void(hdfsFileInfo*, int)> hdfsFreeFileInfo;
            std::function<int(hdfsFS, const char*, int recursive)> hdfsDelete;
            std::function<int(hdfsFS, const char*)> hdfsCreateDirectory;
            std::function<hdfsFileInfo*(hdfsFS, const char*)> hdfsGetPathInfo;
            std::function<int(hdfsFS, const char*, const char*)> hdfsRename;
            std::function<int(hdfsFS, hdfsFile)> hdfsAvailable;
            std::function<int(hdfsFS, const char*, hdfsFS, const char*)> hdfsCopy;
            std::function<int(hdfsFS, const char*, hdfsFS, const char*)> hdfsMove;

        private:
            void LoadAndBind() {
                auto TryLoadAndBind = [this](const char* name, void** handle) -> Status {
                    Status status = LoadLibrary(name, handle);
                    if (!status.ok()) {
                        return status;
                    }

#define BIND_HDFS_FUNC(function)                                            \
                    status = BindFunc(*handle, #function, &function);    \
                    if (!status.ok()) {                                         \
                        return status;                                          \
                    }

                    BIND_HDFS_FUNC(hdfsBuilderConnect);
                    BIND_HDFS_FUNC(hdfsNewBuilder);
                    BIND_HDFS_FUNC(hdfsBuilderSetNameNode);
                    BIND_HDFS_FUNC(hdfsBuilderSetKerbTicketCachePath);
                    BIND_HDFS_FUNC(hdfsCloseFile);
                    BIND_HDFS_FUNC(hdfsPread);
                    BIND_HDFS_FUNC(hdfsWrite);
                    BIND_HDFS_FUNC(hdfsFlush);
                    BIND_HDFS_FUNC(hdfsHSync);
                    BIND_HDFS_FUNC(hdfsOpenFile);
                    BIND_HDFS_FUNC(hdfsExists);
                    BIND_HDFS_FUNC(hdfsListDirectory);
                    BIND_HDFS_FUNC(hdfsFreeFileInfo);
                    BIND_HDFS_FUNC(hdfsDelete);
                    BIND_HDFS_FUNC(hdfsCreateDirectory);
                    BIND_HDFS_FUNC(hdfsGetPathInfo);
                    BIND_HDFS_FUNC(hdfsRename);
                    BIND_HDFS_FUNC(hdfsAvailable);
                    BIND_HDFS_FUNC(hdfsCopy);
                    BIND_HDFS_FUNC(hdfsMove);
#undef BIND_HDFS_FUNC
                    return Status::OK();
            };

                // libhdfs.so won't be in the standard locations. Use the path as specified
                // in the libhdfs documentation.
                char* hdfs_home = getenv("HADOOP_HDFS_HOME");
                if (hdfs_home == nullptr) {
                    status_ = IOError( "Environment variable HADOOP_HDFS_HOME not set", Code::FAILED_PRECONDITION);
                    return;
                }

                std::string path = std::string(hdfs_home) + "/lib/native/libhdfs.so";
                status_ = TryLoadAndBind(path.c_str(), &handle_);
                return;
            }

            Status status_;
            void* handle_ = nullptr;
    };



    class RandomAccessFile {
        public:
            RandomAccessFile(const std::string& fname, LibHDFS* hdfs, hdfsFS fs,
                    hdfsFile file)
                : filename_(fname), hdfs_(hdfs), fs_(fs), file_(file) {}

            ~RandomAccessFile() { hdfs_->hdfsCloseFile(fs_, file_); }

            Status Read(uint64 offset, size_t n, StringPiece* result, char* scratch) const {
                Status s;
                char* dst = scratch;
                while (n > 0 && s.ok()) {
                    tSize r = hdfs_->hdfsPread(fs_, file_, static_cast<tOffset>(offset), dst,
                            static_cast<tSize>(n));
                    if (r > 0) {
                        dst += r;
                        n -= r;
                        offset += r;
                    } else if (r == 0) {
                        s = Status(Code::OUT_OF_RANGE, "Read less bytes than requested");
                    } else if (errno == EINTR || errno == EAGAIN) {
                        // hdfsPread may return EINTR too. Just retry.
                    } else {
                        s = IOError(filename_, errno);
                    }
                }
                *result = StringPiece(scratch, dst - scratch);
                return s;
            }

        private:
            std::string filename_;
            LibHDFS* hdfs_;
            hdfsFS fs_;
            hdfsFile file_;
    };

    class WritableFile {
        public:
            WritableFile(const std::string& fname, LibHDFS* hdfs, hdfsFS fs, hdfsFile file)
                : filename_(fname), hdfs_(hdfs), fs_(fs), file_(file) {}

            ~WritableFile() {
                if (file_ != nullptr) {
                    Close();
                }
            }

            Status Append(const StringPiece& data) {
                if (hdfs_->hdfsWrite(fs_, file_, data.data(),
                            static_cast<tSize>(data.size())) == -1) {
                    return IOError(filename_, errno);
                }
                return Status::OK();
            }

            Status Close() {
                Status result;
                if (hdfs_->hdfsCloseFile(fs_, file_) != 0) {
                    result = IOError(filename_, errno);
                }
                hdfs_ = nullptr;
                fs_ = nullptr;
                file_ = nullptr;
                return result;
            }

            Status Flush() {
                if (hdfs_->hdfsFlush(fs_, file_) != 0) {
                    return IOError(filename_, errno);
                }
                return Status::OK();
            }

            Status Sync() {
                if (hdfs_->hdfsHSync(fs_, file_) != 0) {
                    return IOError(filename_, errno);
                }
                return Status::OK();
            }

        private:
            std::string filename_;
            LibHDFS* hdfs_;
            hdfsFS fs_;
            hdfsFile file_;
    };


    class HadoopFileSystem {
        public:
            HadoopFileSystem();
            ~HadoopFileSystem();

            Status NewRandomAccessFile(const std::string& fname,
                                    std::shared_ptr<RandomAccessFile>* result);

            Status NewWritableFile(const std::string& fname,
                                std::shared_ptr<WritableFile>* result);

            Status NewAppendableFile(const std::string& fname,
                                    std::shared_ptr<WritableFile>* result);

            Status FileExists(const std::string& fname);

            Status GetChildren(const std::string& dir, std::vector<std::string>* result);

            Status DeleteFile(const std::string& fname);

            Status CreateDir(const std::string& name);

            Status DeleteDir(const std::string& name);

            Status GetFileSize(const std::string& fname, uint64* size);

            Status RenameFile(const std::string& src, const std::string& target);

            Status Stat(const std::string& fname, FileStatistics* stat);

            std::string TranslateName(const std::string& name) const;

            Status CopyToLocal(const std::string& src, const std::string& dst);
            Status CopyToRemote(const std::string& src, const std::string& dst);

            Status MoveToLocal(const std::string& src, const std::string& dst);
            Status MoveToRemote(const std::string& src, const std::string& dst);

            Status Connect(StringPiece fname, hdfsFS* fs);

            LibHDFS* hdfs_;
    };

}  // namespace

#endif  // HADOOP_FILE_SYSTEM_H_
