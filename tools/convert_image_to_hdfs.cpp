// This program converts a set of images to a lmdb/leveldb by storing them
// as Datum proto buffers.
// Usage:
//   convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME
//
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   subfolder1/file1.JPEG 7
//   ....

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

#include "caffe/hdfs/hadoop_file_system.h"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;

DEFINE_bool(gray, false,
        "When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, false,
        "Randomly shuffle the order of images and their labels");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");
DEFINE_bool(check_size, true,
        "When this option is on, check that all the datum have the same size");
DEFINE_bool(encoded, false,
        "When this option is on, the encoded image will be save in datum");
DEFINE_string(encode_type, "",
        "Optional: What type should we encode the image as ('png','jpg',...).");

int main(int argc, char** argv) {
#ifdef USE_OPENCV
    ::google::InitGoogleLogging(argv[0]);
    // Print output to stderr (while still logging)
    FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
    namespace gflags = google;
#endif

    gflags::SetUsageMessage("Convert a set of images to the hdfs\n"
            "format used as input for Caffe.\n"
            "Usage:\n"
            "    convert_image_to_hdfs [FLAGS] ROOTFOLDER/ LISTFILE HDFSPATH\n");
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    if (argc < 4) {
        gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_image_to_hdfs");
        return 1;
    }

    const bool is_color = !FLAGS_gray;
    const bool check_size = FLAGS_check_size;
    const bool encoded = FLAGS_encoded;
    const string encode_type = FLAGS_encode_type;
    int datum_len = 0;

    std::ifstream infile(argv[2]);
    std::vector<std::pair<std::string, int> > lines;
    std::string line;
    size_t pos;
    int label;
    while (std::getline(infile, line)) {
        pos = line.find_last_of(' ');
        label = atoi(line.substr(pos + 1).c_str());
        lines.push_back(std::make_pair(line.substr(0, pos), label));
    }
    if (FLAGS_shuffle) {
        // randomly shuffle data
        LOG(INFO) << "Shuffling data";
        shuffle(lines.begin(), lines.end());
    }
    LOG(INFO) << "A total of " << lines.size() << " images.";

    if (encode_type.size() && !encoded)
        LOG(INFO) << "encode_type specified, assuming encoded=true.";

    int resize_height = std::max<int>(0, FLAGS_resize_height);
    int resize_width = std::max<int>(0, FLAGS_resize_width);

    StringPiece filename = StringPiece(argv[3]);
    CHECK_EQ(filename.starts_with("hdfs://"), true) << "dst path must starts with 'hdfs://'";
    HadoopFileSystem hdfs;
    std::shared_ptr<WritableFile> wf;
    Status status1 = hdfs.NewWritableFile(filename.ToString(), &wf);
    CHECK_EQ(status1.ok(), true) << "can't open file: " << status1;

    // Storing to db
    std::string root_folder(argv[1]);
    Datum datum;
    int count = 0;
    int data_size = 0;
    bool data_size_initialized = false;

    for (int line_id = 0; line_id < lines.size(); ++line_id) {
        bool status;
        std::string enc = encode_type;
        if (encoded && !enc.size()) {
            // Guess the encoding type from the file name
            string fn = lines[line_id].first;
            size_t p = fn.rfind('.');
            if ( p == fn.npos )
                LOG(WARNING) << "Failed to guess the encoding of '" << fn << "'";
            enc = fn.substr(p);
            std::transform(enc.begin(), enc.end(), enc.begin(), ::tolower);
        }
        status = ReadImageToDatum(root_folder + lines[line_id].first,
                lines[line_id].second, resize_height, resize_width, is_color,
                enc, &datum);
        if (status == false) continue;
        if (check_size) {
            if (!data_size_initialized) {
                data_size = datum.channels() * datum.height() * datum.width();
                data_size_initialized = true;
            } else {
                const std::string& data = datum.data();
                CHECK_EQ(data.size(), data_size) << "Incorrect data field size "
                    << data.size();
            }
        }

        string out;
        CHECK(datum.SerializeToString(&out));
        datum_len = out.length();
        Status status2 = wf->Append(StringPiece(out));
        CHECK_EQ(status2.ok(), true) << "can't write file: " << status2;

        if (++count % 1000 == 0) {
            LOG(INFO) << "Processed " << count << " files.";
        }
    }

    // write the last batch
    if (count % 1000 != 0) {
        LOG(INFO) << "Processed " << count << " files.";
    }

    wf->Close();
    LOG(INFO) << "The datum length is " << datum_len << " ,you must specify it in hdfs layer";
#else
    LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
    return 0;
}
