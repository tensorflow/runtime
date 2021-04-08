// Copyright 2020 The TensorFlow Runtime Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This file defines error utils.

#include "tfrt/support/error_util.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/logging.h"
#include "tfrt/support/string_util.h"

namespace tfrt {
namespace {
void LogIfErrorImpl(Error error, Severity severity) {
  llvm::handleAllErrors(std::move(error), [&](const llvm::ErrorInfoBase& info) {
    tfrt::internal::LogStream(__FILE__, __LINE__, severity) << info.message();
  });
}
}  // namespace

void LogIfError(Error&& error) {
  LogIfErrorImpl(std::move(error), Severity::ERROR);
}

void DieIfError(Error&& error) {
  LogIfErrorImpl(std::move(error), Severity::FATAL);
}

string_view ErrorName(ErrorCode code) {
  switch (code) {
#define ERROR_TYPE(ENUM)   \
  case ErrorCode::k##ENUM: \
    return #ENUM;
#include "tfrt/support/error_type.def"  // NOLINT
  }
}

char BaseTypedErrorInfo::ID;
char ErrorCollection::ID;

void ErrorCollection::AddError(Error error) {
  if (error) {
    DieIfError(llvm::handleErrors(
        std::move(error),
        [&](std::unique_ptr<BaseTypedErrorInfo> ei) {
          errors_.push_back(std::move(ei));
        },
        [&](std::unique_ptr<ErrorCollection> ei) {
          errors_.insert(errors_.end(),
                         std::make_move_iterator(ei->errors_.begin()),
                         std::make_move_iterator(ei->errors_.end()));
        }));
  }
}

const llvm::SmallVector<std::unique_ptr<BaseTypedErrorInfo>, 4>&
ErrorCollection::GetAllErrors() const {
  return std::move(errors_);
}

void ErrorCollection::log(raw_ostream& OS) const {
  if (errors_.empty()) {
    OS << llvm::toString(Error::success());
    return;
  }
  if (errors_.size() == 1) {
    errors_[0]->log(OS);
    return;
  }

  llvm::SmallVector<std::string, 4> msg;
  msg.reserve(1 + errors_.size());
  msg.push_back(StrCat("Found ", errors_.size(), " errors:"));
  int index = 0;
  for (const auto& e : errors_) {
    std::string str;
    llvm::raw_string_ostream os(str);
    os << "  (" << ++index << ") ";
    e->log(os);
    msg.push_back(str);
  }
  OS << Join(msg.begin(), msg.end(), "\n");
}

}  // namespace tfrt
