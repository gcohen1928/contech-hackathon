"use client";

import * as React from "react";

import { NavDocuments } from "@/components/NavDocuments";
import { NavDatabases } from "@/components/NavDatabases";
import { Sidebar, SidebarContent } from "@/components/ui/sidebar";
import { DATABASES, DOCUMENTS } from "@/state/selectedFile";

// This is sample data.
const FileExplorer = ({ ...props }: React.ComponentProps<typeof Sidebar>) => {
  return (
    <Sidebar {...props}>
      <SidebarContent>
        <NavDocuments items={DOCUMENTS} />
        <NavDatabases databases={DATABASES} />
      </SidebarContent>
    </Sidebar>
  );
};

export default FileExplorer;
