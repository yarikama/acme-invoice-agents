import {
  ChartLineUpIcon,
  CheckCircleIcon,
  CircleNotchIcon,
  CreditCardIcon,
  FileArrowDownIcon,
  FileTextIcon,
  FlagIcon,
  FlowArrowIcon,
  GavelIcon,
  HourglassMediumIcon,
  type Icon,
  ListBulletsIcon,
  ListChecksIcon,
  MagnifyingGlassIcon,
  PencilSimpleLineIcon,
  ProhibitIcon,
  PulseIcon,
  ReceiptIcon,
  TrayIcon,
  UserCheckIcon,
  WarningOctagonIcon,
} from "@phosphor-icons/react";

import type { Segment } from "../state/pipeline";
import type { Filter } from "./filters";

// One icon per category, used everywhere that category appears.
export const STATUS_ICON: Record<Filter, Icon> = {
  all: TrayIcon,
  awaiting_vp: HourglassMediumIcon,
  active: CircleNotchIcon,
  paid: CheckCircleIcon,
  rejected: ProhibitIcon,
  failed: WarningOctagonIcon,
};

export const STAGE_ICON: Record<Segment["key"], Icon> = {
  ingest: FileArrowDownIcon,
  validate: ListChecksIcon,
  draft: PencilSimpleLineIcon,
  critique: MagnifyingGlassIcon,
  vp: UserCheckIcon,
  pay: CreditCardIcon,
};

export const SECTION_ICON = {
  invoices: ReceiptIcon,
  summary: FileTextIcon,
  decision: GavelIcon,
  stream: PulseIcon,
  findings: FlagIcon,
  lineItems: ListBulletsIcon,
  routing: FlowArrowIcon,
  throughput: ChartLineUpIcon,
} satisfies Record<string, Icon>;
