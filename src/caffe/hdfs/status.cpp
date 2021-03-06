/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "caffe/hdfs/status.h"
#include <stdio.h>

namespace caffe {

    Status::Status(Code code, StringPiece msg) {
        assert(code != Code::OK);
        state_ = new State;
        state_->code = code;
        state_->msg = msg.ToString();
    }

    void Status::Update(const Status& new_status) {
        if (ok()) {
            *this = new_status;
        }
    }

    void Status::SlowCopyFrom(const State* src) {
        delete state_;
        if (src == nullptr) {
            state_ = nullptr;
        } else {
            state_ = new State(*src);
        }
    }

    const std::string& Status::empty_string() {
        static std::string* empty = new std::string;
        return *empty;
    }

    std::string Status::ToString() const {
        if (state_ == NULL) {
            return "OK";
        } else {
            char tmp[30];
            const char* type;
            switch (code()) {
                case Code::CANCELLED:
                    type = "Cancelled";
                    break;
                case Code::UNKNOWN:
                    type = "Unknown";
                    break;
                case Code::INVALID_ARGUMENT:
                    type = "Invalid argument";
                    break;
                case Code::DEADLINE_EXCEEDED:
                    type = "Deadline exceeded";
                    break;
                case Code::NOT_FOUND:
                    type = "Not found";
                    break;
                case Code::ALREADY_EXISTS:
                    type = "Already exists";
                    break;
                case Code::PERMISSION_DENIED:
                    type = "Permission denied";
                    break;
                case Code::UNAUTHENTICATED:
                    type = "Unauthenticated";
                    break;
                case Code::RESOURCE_EXHAUSTED:
                    type = "Resource exhausted";
                    break;
                case Code::FAILED_PRECONDITION:
                    type = "Failed precondition";
                    break;
                case Code::ABORTED:
                    type = "Aborted";
                    break;
                case Code::OUT_OF_RANGE:
                    type = "Out of range";
                    break;
                case Code::UNIMPLEMENTED:
                    type = "Unimplemented";
                    break;
                case Code::INTERNAL:
                    type = "Internal";
                    break;
                case Code::UNAVAILABLE:
                    type = "Unavailable";
                    break;
                case Code::DATA_LOSS:
                    type = "Data loss";
                    break;
                default:
                    snprintf(tmp, sizeof(tmp), "Unknown code(%d)",
                            static_cast<int>(code()));
                    type = tmp;
                    break;
            }
            std::string result(type);
            result += ": ";
            result += state_->msg;
            return result;
        }
    }

    std::ostream& operator<<(std::ostream& os, const Status& x) {
        os << x.ToString();
        return os;
    }

}  // namespace
