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

#include <errno.h>
#include <dlfcn.h>

#include "caffe/hdfs/hadoop_file_system.h"

namespace caffe {

#define DECLARE_ERROR(FUNC, CONST)                  \
    Status FUNC(StringPiece args) {                 \
        return Status(Code::CONST, args);           \
    }

    DECLARE_ERROR(Cancelled, CANCELLED)
    DECLARE_ERROR(InvalidArgument, INVALID_ARGUMENT)
    DECLARE_ERROR(NotFound, NOT_FOUND)
    DECLARE_ERROR(AlreadyExists, ALREADY_EXISTS)
    DECLARE_ERROR(ResourceExhausted, RESOURCE_EXHAUSTED)
    DECLARE_ERROR(Unavailable, UNAVAILABLE)
    DECLARE_ERROR(FailedPrecondition, FAILED_PRECONDITION)
    DECLARE_ERROR(OutOfRange, OUT_OF_RANGE)
    DECLARE_ERROR(Unimplemented, UNIMPLEMENTED)
    DECLARE_ERROR(Internal, INTERNAL)
    DECLARE_ERROR(Aborted, ABORTED)
    DECLARE_ERROR(DeadlineExceeded, DEADLINE_EXCEEDED)
    DECLARE_ERROR(DataLoss, DATA_LOSS)
    DECLARE_ERROR(Unknown, UNKNOWN)
    DECLARE_ERROR(PermissionDenied, PERMISSION_DENIED)
    DECLARE_ERROR(Unauthenticated, UNAUTHENTICATED)

#undef DECLARE_ERROR

    Status LibHDFS::LoadLibrary(const char* library_filename, void** handle){
        *handle = dlopen(library_filename, RTLD_NOW | RTLD_LOCAL);
        if (!*handle) {
            return IOError(dlerror(), Code::NOT_FOUND);
        }

        return Status::OK();
    }

    Status LibHDFS::GetSymbolFromLibrary(void* handle, const char* symbo_name, void** symbol) {
        *symbol = dlsym(handle, symbo_name);
        if (!*symbol) {
            return IOError(dlerror(), Code::NOT_FOUND);
        }
        return Status::OK();
    }

    template <typename R, typename... Args>
        Status LibHDFS::BindFunc(void* handle, const char* name, std::function<R(Args...)>* func) {
            void* symbol_ptr = nullptr;
            Status status = GetSymbolFromLibrary(handle, name, &symbol_ptr);
            if (!status.ok()) {
                return status;
            }
            *func = reinterpret_cast<R (*)(Args...)>(symbol_ptr);
            return Status::OK();
        }


    HadoopFileSystem::HadoopFileSystem() : hdfs_(LibHDFS::Load()) {}

    HadoopFileSystem::~HadoopFileSystem() {}

    // We rely on HDFS connection caching here. The HDFS client calls
    // org.apache.hadoop.fs.FileSystem.get(), which caches the connection
    // internally.
    Status HadoopFileSystem::Connect(StringPiece fname, hdfsFS* fs) {
        Status status = hdfs_->status();
        if (!status.ok()) {
            return status;
        }

        StringPiece scheme, namenode, path;
        ParseURI(fname, &scheme, &namenode, &path);
        const std::string nn = namenode.ToString();

        hdfsBuilder* builder = hdfs_->hdfsNewBuilder();
        if (scheme == "file") {
            hdfs_->hdfsBuilderSetNameNode(builder, nullptr);
        } else if (scheme == "hdfs") {
            hdfs_->hdfsBuilderSetNameNode(builder, nn.c_str());
        } else {
            return InvalidArgument(scheme.ToString() + "scheme must be file or hdfs");
        }

        char* ticket_cache_path = getenv("KERB_TICKET_CACHE_PATH");
        if (ticket_cache_path != nullptr) {
            hdfs_->hdfsBuilderSetKerbTicketCachePath(builder, ticket_cache_path);
        }
        *fs = hdfs_->hdfsBuilderConnect(builder);
        if (*fs == nullptr) {
            return NotFound(strerror(errno));
        }
        return Status::OK();
    }

    std::string HadoopFileSystem::TranslateName(const std::string& name) const {
        StringPiece scheme, namenode, path;
        ParseURI(name, &scheme, &namenode, &path);
        return path.ToString();
    }

    Status HadoopFileSystem::NewRandomAccessFile(
            const std::string& fname, std::shared_ptr<RandomAccessFile>* result) {
        hdfsFS fs = nullptr;
        Status status = Connect(fname, &fs);
        if (!status.ok()) {
            return status;
        }

        hdfsFile file =
            hdfs_->hdfsOpenFile(fs, TranslateName(fname).c_str(), O_RDONLY, 0, 0, 0);
        if (file == nullptr) {
            return IOError(fname, errno);
        }
        result->reset(new RandomAccessFile(fname, hdfs_, fs, file));
        return Status::OK();
    }

    Status HadoopFileSystem::NewWritableFile(
            const std::string& fname, std::shared_ptr<WritableFile>* result) {
        hdfsFS fs = nullptr;
        Status status = Connect(fname, &fs);
        if (!status.ok()) {
            return status;
        }

        hdfsFile file =
            hdfs_->hdfsOpenFile(fs, TranslateName(fname).c_str(), O_WRONLY, 0, 0, 0);
        if (file == nullptr) {
            return IOError(fname, errno);
        }
        result->reset(new WritableFile(fname, hdfs_, fs, file));
        return Status::OK();
    }

    Status HadoopFileSystem::NewAppendableFile(
            const std::string& fname, std::shared_ptr<WritableFile>* result) {
        hdfsFS fs = nullptr;
        Status status = Connect(fname, &fs);
        if (!status.ok()) {
            return status;
        }

        hdfsFile file = hdfs_->hdfsOpenFile(fs, TranslateName(fname).c_str(),
                O_WRONLY | O_APPEND, 0, 0, 0);
        if (file == nullptr) {
            return IOError(fname, errno);
        }
        result->reset(new WritableFile(fname, hdfs_, fs, file));
        return Status::OK();
    }

    Status HadoopFileSystem::FileExists(const std::string& fname) {
        hdfsFS fs = nullptr;
        Status status = Connect(fname, &fs);
        if (!status.ok()) {
            return status;
        }

        if (hdfs_->hdfsExists(fs, TranslateName(fname).c_str()) == 0) {
            return Status::OK();
        }
        return NotFound(fname);
    }

    Status HadoopFileSystem::GetChildren(const std::string& dir,
            std::vector<std::string>* result) {
        result->clear();
        hdfsFS fs = nullptr;
        Status status = Connect(dir, &fs);
        if (!status.ok()) {
            return status;
        }

        // hdfsListDirectory returns nullptr if the directory is empty. Do a separate
        // check to verify the directory exists first.
        FileStatistics stat;
        status = Stat(dir, &stat);
        if (!status.ok()) {
            return status;
        }

        int entries = 0;
        hdfsFileInfo* info =
            hdfs_->hdfsListDirectory(fs, TranslateName(dir).c_str(), &entries);
        if (info == nullptr) {
            if (stat.is_directory) {
                // Assume it's an empty directory.
                return Status::OK();
            }
            return IOError(dir, errno);
        }
        for (int i = 0; i < entries; i++) {
            result->push_back(Basename(info[i].mName).ToString());
        }
        hdfs_->hdfsFreeFileInfo(info, entries);
        return Status::OK();
    }

    Status HadoopFileSystem::DeleteFile(const std::string& fname) {
        hdfsFS fs = nullptr;
        Status status = Connect(fname, &fs);
        if (!status.ok()) {
            return status;
        }

        if (hdfs_->hdfsDelete(fs, TranslateName(fname).c_str(),
                    /*recursive=*/0) != 0) {
            return IOError(fname, errno);
        }
        return Status::OK();
    }

    Status HadoopFileSystem::CreateDir(const std::string& dir) {
        hdfsFS fs = nullptr;
        Status status = Connect(dir, &fs);
        if (!status.ok()) {
            return status;
        }

        if (hdfs_->hdfsCreateDirectory(fs, TranslateName(dir).c_str()) != 0) {
            return IOError(dir, errno);
        }
        return Status::OK();
    }

    Status HadoopFileSystem::DeleteDir(const std::string& dir) {
        hdfsFS fs = nullptr;
        Status status = Connect(dir, &fs);
        if (!status.ok()) {
            return status;
        }

        // Count the number of entries in the directory, and only delete if it's
        // non-empty. This is consistent with the interface, but note that there's
        // a race condition where a file may be added after this check, in which
        // case the directory will still be deleted.
        int entries = 0;
        hdfsFileInfo* info =
            hdfs_->hdfsListDirectory(fs, TranslateName(dir).c_str(), &entries);
        if (info != nullptr) {
            hdfs_->hdfsFreeFileInfo(info, entries);
        }
        // Due to HDFS bug HDFS-8407, we can't distinguish between an error and empty
        // folder, expscially for Kerberos enable setup, EAGAIN is quite common when
        // the call is actually successful. Check again by Stat.
        if (info == nullptr && errno != 0) {
            FileStatistics stat;
            Status status = Stat(dir, &stat);
            if (!status.ok()) {
                return status;
            }

        }

        if (entries > 0) {
            return FailedPrecondition("Cannot delete a non-empty directory.");
        }
        if (hdfs_->hdfsDelete(fs, TranslateName(dir).c_str(),
                    /*recursive=*/1) != 0) {
            return IOError(dir, errno);
        }
        return Status::OK();
    }

    Status HadoopFileSystem::GetFileSize(const std::string& fname, uint64* size) {
        hdfsFS fs = nullptr;
        Status status = Connect(fname, &fs);
        if (!status.ok()) {
            return status;
        }

        hdfsFileInfo* info = hdfs_->hdfsGetPathInfo(fs, TranslateName(fname).c_str());
        if (info == nullptr) {
            return IOError(fname, errno);
        }
        *size = static_cast<uint64>(info->mSize);
        hdfs_->hdfsFreeFileInfo(info, 1);
        return Status::OK();
    }

    Status HadoopFileSystem::RenameFile(const std::string& src, const std::string& target) {
        hdfsFS fs = nullptr;
        Status status = Connect(src, &fs);
        if (!status.ok()) {
            return status;
        }

        if (hdfs_->hdfsExists(fs, TranslateName(target).c_str()) == 0 &&
                hdfs_->hdfsDelete(fs, TranslateName(target).c_str(),
                    /*recursive=*/0) != 0) {
            return IOError(target, errno);
        }

        if (hdfs_->hdfsRename(fs, TranslateName(src).c_str(),
                    TranslateName(target).c_str()) != 0) {
            return IOError(src, errno);
        }
        return Status::OK();
    }

    Status HadoopFileSystem::Stat(const std::string& fname, FileStatistics* stats) {
        hdfsFS fs = nullptr;
        Status status = Connect(fname, &fs);
        if (!status.ok()) {
            return status;
        }

        hdfsFileInfo* info = hdfs_->hdfsGetPathInfo(fs, TranslateName(fname).c_str());
        if (info == nullptr) {
            return IOError(fname, errno);
        }
        stats->length = static_cast<int64>(info->mSize);
        stats->mtime_nsec = static_cast<int64>(info->mLastMod) * 1e9;
        stats->is_directory = info->mKind == kObjectKindDirectory;
        hdfs_->hdfsFreeFileInfo(info, 1);
        return Status::OK();
    }

    Status HadoopFileSystem::CopyToLocal(const std::string& src, const std::string& dst) {
        Status status;
        hdfsFS fs = nullptr;
        status = Connect(src, &fs);
        if (!status.ok()) {
            return status;
        }

        hdfsFS lfs = nullptr;
        status = Connect(dst, &lfs);
        if (!status.ok()) {
            return status;
        }

        if(hdfs_->hdfsCopy(fs, src.c_str(), lfs, dst.c_str()) != 0) {
            return IOError("from " + src + " to " + dst, errno);
        }

        return Status::OK();
    }

    Status HadoopFileSystem::CopyToRemote(const std::string& src, const std::string& dst) {
        Status status;
        hdfsFS lfs = nullptr;
        status = Connect(src, &lfs);
        if (!status.ok()) {
            return status;
        }

        hdfsFS fs = nullptr;
        status = Connect(dst, &fs);
        if (!status.ok()) {
            return status;
        }

        if(hdfs_->hdfsCopy(lfs, src.c_str(), fs, dst.c_str()) != 0) {
            return IOError("from " + src + " to " + dst, errno);
        }

        return Status::OK();

    }

    Status HadoopFileSystem::MoveToLocal(const std::string& src, const std::string& dst) {
        Status status;
        hdfsFS fs = nullptr;
        status = Connect(src, &fs);
        if (!status.ok()) {
            return status;
        }

        hdfsFS lfs = nullptr;
        status = Connect(dst, &lfs);
        if (!status.ok()) {
            return status;
        }

        if(hdfs_->hdfsMove(fs, src.c_str(), lfs, dst.c_str()) != 0) {
            return IOError("from " + src + " to " + dst, errno);
        }

        return Status::OK();
    }

    Status HadoopFileSystem::MoveToRemote(const std::string& src, const std::string& dst) {
        Status status;
        hdfsFS lfs = nullptr;
        status = Connect(src, &lfs);
        if (!status.ok()) {
            return status;
        }

        hdfsFS fs = nullptr;
        status = Connect(dst, &fs);
        if (!status.ok()) {
            return status;
        }

        if(hdfs_->hdfsMove(lfs, src.c_str(), fs, dst.c_str()) != 0) {
            return IOError("from " + src + " to " + dst, errno);
        }

        return Status::OK();

    }


}  // namespace
