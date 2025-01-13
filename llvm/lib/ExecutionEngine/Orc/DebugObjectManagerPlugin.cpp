//===------- DebugObjectManagerPlugin.cpp - JITLink debug objects ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// FIXME: Update Plugin to poke the debug object into a new JITLink section,
//        rather than creating a new allocation.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/DebugObjectManagerPlugin.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/ExecutionEngine/JITLink/JITLinkDylib.h"
#include "llvm/ExecutionEngine/JITLink/JITLinkMemoryManager.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/MSVCErrorWorkarounds.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"

#include <set>

#define DEBUG_TYPE "orc"

using namespace llvm::jitlink;
using namespace llvm::object;

namespace llvm {
namespace orc {

class DebugObjectSection {
public:
  virtual void setTargetMemoryRange(SectionRange Range) = 0;
  virtual void dump(raw_ostream &OS, StringRef Name) {}
  virtual ~DebugObjectSection() = default;
};

enum DebugObjectFlags : int {
  // Request final target memory load-addresses for all sections.
  ReportFinalSectionLoadAddresses = 1 << 0,

  // We found sections with debug information when processing the input object.
  HasDebugSections = 1 << 1,
};

/// The plugin creates a debug object from when JITLink starts processing the
/// corresponding LinkGraph. It provides access to the pass configuration of
/// the LinkGraph and calls the finalization function, once the resulting link
/// artifact was emitted.
///
class DebugObject {
public:
  DebugObject(JITLinkMemoryManager &MemMgr, const JITLinkDylib *JD,
              ExecutionSession &ES)
      : MemMgr(MemMgr), JD(JD), ES(ES), Flags(DebugObjectFlags{}) {}

  bool hasFlags(DebugObjectFlags F) const { return Flags & F; }
  void setFlags(DebugObjectFlags F) {
    Flags = static_cast<DebugObjectFlags>(Flags | F);
  }
  void clearFlags(DebugObjectFlags F) {
    Flags = static_cast<DebugObjectFlags>(Flags & ~F);
  }

  using FinalizeContinuation = std::function<void(Expected<ExecutorAddrRange>)>;

  void finalizeAsync(FinalizeContinuation OnFinalize);

  virtual ~DebugObject() {
    if (Alloc) {
      std::vector<FinalizedAlloc> Allocs;
      Allocs.push_back(std::move(Alloc));
      if (Error Err = MemMgr.deallocate(std::move(Allocs)))
        ES.reportError(std::move(Err));
    }
  }

  virtual void reportSectionTargetMemoryRange(StringRef Name,
                                              SectionRange TargetMem) {}

protected:
  using InFlightAlloc = JITLinkMemoryManager::InFlightAlloc;
  using FinalizedAlloc = JITLinkMemoryManager::FinalizedAlloc;

  virtual Expected<SimpleSegmentAlloc> finalizeWorkingMemory() = 0;

  JITLinkMemoryManager &MemMgr;
  const JITLinkDylib *JD = nullptr;

private:
  ExecutionSession &ES;
  DebugObjectFlags Flags;
  FinalizedAlloc Alloc;
};

// Finalize working memory and take ownership of the resulting allocation. Start
// copying memory over to the target and pass on the result once we're done.
// Ownership of the allocation remains with us for the rest of our lifetime.
void DebugObject::finalizeAsync(FinalizeContinuation OnFinalize) {
  assert(!Alloc && "Cannot finalize more than once");

  if (auto SimpleSegAlloc = finalizeWorkingMemory()) {
    auto ROSeg = SimpleSegAlloc->getSegInfo(MemProt::Read);
    ExecutorAddrRange DebugObjRange(ROSeg.Addr, ROSeg.WorkingMem.size());
    SimpleSegAlloc->finalize(
        [this, DebugObjRange,
         OnFinalize = std::move(OnFinalize)](Expected<FinalizedAlloc> FA) {
          if (FA) {
            Alloc = std::move(*FA);
            OnFinalize(DebugObjRange);
          } else
            OnFinalize(FA.takeError());
        });
  } else
    OnFinalize(SimpleSegAlloc.takeError());
}

static const std::set<StringRef> DwarfSectionNames = {
#define HANDLE_DWARF_SECTION(ENUM_NAME, ELF_NAME, CMDLINE_NAME, OPTION)        \
  ELF_NAME,
#include "llvm/BinaryFormat/Dwarf.def"
#undef HANDLE_DWARF_SECTION
};

static bool isDwarfSection(StringRef SectionName) {
  return DwarfSectionNames.count(SectionName) == 1;
}

template <typename ELFT> Error fixUp(StringRef Buffer, LinkGraph &G) {
    if (auto *GraphSec = G.findSectionByName(*Name))
    Header->sh_addr =
     static_cast<typename ELFT::uint>(SectionRange(*GraphSec).getStart().getValue());
}

DebugObjectManagerPlugin::DebugObjectManagerPlugin(
    ExecutionSession &ES, std::unique_ptr<DebugObjectRegistrar> Target,
    bool RequireDebugSections, bool AutoRegisterCode)
    : ES(ES), Target(std::move(Target)),
      RequireDebugSections(RequireDebugSections),
      AutoRegisterCode(AutoRegisterCode) {}

DebugObjectManagerPlugin::DebugObjectManagerPlugin(
    ExecutionSession &ES, std::unique_ptr<DebugObjectRegistrar> Target)
    : DebugObjectManagerPlugin(ES, std::move(Target), true, true) {}

DebugObjectManagerPlugin::~DebugObjectManagerPlugin() = default;

Error DebugObjectManagerPlugin::fixUpDebugObject(LinkGraph &G) {
  auto *DebugObjSec = G.findSectionByName(".jitlink_original_object_content");
  assert(DebugObjSec && "No ELF debug object section?");
  assert(DebugObjSec.blocks_size() == 1 && "ELF debug object contains multiple blocks?");
  auto DebugObjContent = (*DebugObjSec.blocks_begin())->getAlreadyMutableContent();
  StringRef DebugObj(DebugObjContent.data(), DebugObjContent.size());

  unsigned char Class, Endian;
  std::tie(Class, Endian) = getElfArchType(DebugObj);
  if (Class == ELF::ELFCLASS32) {
    if (Endian == ELF::ELFDATA2LSB)
      return fixUp<ELF32LE>(DebugObj, G);
    else if (Endian == ELF::ELFDATA2MSB)
      return fixUp<ELF32BE>(DebugObj, G);
  } else if (Class == ELF::ELFCLASS64) {
    if (Endian == ELF::ELFDATA2LSB)
      return fixUp<ELF64LE>(DebugObj, G);
    else if (Endian == ELF::ELFDATA2MSB)
      return fixUp<ELF64BE>(DebugObj, G);
  }
  // Unsupported combo. Remove the debug object section.
  G.removeSection(*DebugObjSec);
  LLVM_DEBUG({
    dbgs() << "Can't emit debug object for " << G.getName()
           << ": Unsupported ELF class / endianness.\n";
  });
  return Error::success();
}                                                                 

void DebugObjectManagerPlugin::modifyPassConfig(
    MaterializationResponsibility &MR, LinkGraph &G,
    PassConfiguration &PassConfig) {
  // Not all link artifacts have associated debug objects.
  std::lock_guard<std::mutex> Lock(PendingObjsLock);
  auto It = PendingObjs.find(&MR);
  if (It == PendingObjs.end())
    return;

  DebugObject &DebugObj = *It->second;
  if (DebugObj.hasFlags(ReportFinalSectionLoadAddresses)) {
    PassConfig.PostAllocationPasses.push_back(
        [&DebugObj](LinkGraph &Graph) -> Error {
          for (const Section &GraphSection : Graph.sections())
            DebugObj.reportSectionTargetMemoryRange(GraphSection.getName(),
                                                    SectionRange(GraphSection));
          return Error::success();
        });
  }
}

Error DebugObjectManagerPlugin::notifyEmitted(
    MaterializationResponsibility &MR) {
  std::lock_guard<std::mutex> Lock(PendingObjsLock);
  auto It = PendingObjs.find(&MR);
  if (It == PendingObjs.end())
    return Error::success();

  // During finalization the debug object is registered with the target.
  // Materialization must wait for this process to finish. Otherwise we might
  // start running code before the debugger processed the corresponding debug
  // info.
  std::promise<MSVCPError> FinalizePromise;
  std::future<MSVCPError> FinalizeErr = FinalizePromise.get_future();

  It->second->finalizeAsync(
      [this, &FinalizePromise, &MR](Expected<ExecutorAddrRange> TargetMem) {
        // Any failure here will fail materialization.
        if (!TargetMem) {
          FinalizePromise.set_value(TargetMem.takeError());
          return;
        }
        if (Error Err =
                Target->registerDebugObject(*TargetMem, AutoRegisterCode)) {
          FinalizePromise.set_value(std::move(Err));
          return;
        }

        // Once our tracking info is updated, notifyEmitted() can return and
        // finish materialization.
        FinalizePromise.set_value(MR.withResourceKeyDo([&](ResourceKey K) {
          assert(PendingObjs.count(&MR) && "We still hold PendingObjsLock");
          std::lock_guard<std::mutex> Lock(RegisteredObjsLock);
          RegisteredObjs[K].push_back(std::move(PendingObjs[&MR]));
          PendingObjs.erase(&MR);
        }));
      });

  return FinalizeErr.get();
}

Error DebugObjectManagerPlugin::notifyFailed(
    MaterializationResponsibility &MR) {
  std::lock_guard<std::mutex> Lock(PendingObjsLock);
  PendingObjs.erase(&MR);
  return Error::success();
}

void DebugObjectManagerPlugin::notifyTransferringResources(JITDylib &JD,
                                                           ResourceKey DstKey,
                                                           ResourceKey SrcKey) {
  // Debug objects are stored by ResourceKey only after registration.
  // Thus, pending objects don't need to be updated here.
  std::lock_guard<std::mutex> Lock(RegisteredObjsLock);
  auto SrcIt = RegisteredObjs.find(SrcKey);
  if (SrcIt != RegisteredObjs.end()) {
    // Resources from distinct MaterializationResponsibilitys can get merged
    // after emission, so we can have multiple debug objects per resource key.
    for (std::unique_ptr<DebugObject> &DebugObj : SrcIt->second)
      RegisteredObjs[DstKey].push_back(std::move(DebugObj));
    RegisteredObjs.erase(SrcIt);
  }
}

Error DebugObjectManagerPlugin::notifyRemovingResources(JITDylib &JD,
                                                        ResourceKey Key) {
  // Removing the resource for a pending object fails materialization, so they
  // get cleaned up in the notifyFailed() handler.
  std::lock_guard<std::mutex> Lock(RegisteredObjsLock);
  RegisteredObjs.erase(Key);

  // TODO: Implement unregister notifications.
  return Error::success();
}

} // namespace orc
} // namespace llvm
