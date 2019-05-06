% Proposal Class file
% (c) Matthieu Bloch 2017
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{proposal}[2017/09/03 Proposal, v1.0]
% History
% 09/03/17 First version

\LoadClass[11pt]{article}% inherits from base class article

\newif\ifgaramond
\garamondfalse
\newif\iftimes
\timesfalse

\RequirePackage{amsmath}
\RequirePackage{array}
\RequirePackage{amssymb}
\RequirePackage{amsbsy}
\RequirePackage{graphicx}
\RequirePackage{wrapfig}
\RequirePackage{caption}
\RequirePackage{subcaption}
\RequirePackage{multirow}
\RequirePackage{url}
\RequirePackage{changebar}
\RequirePackage{cite}
\RequirePackage{bbding}
\RequirePackage{bbm}
\RequirePackage{stmaryrd}
%\RequirePackage{subfloat}
\RequirePackage{acronym}
\RequirePackage{color,colortbl}
\RequirePackage{mdframed}
% \RequirePackage{titlesec} % to deal with title spacing
\RequirePackage[colorlinks]{hyperref} % may have to remove
\RequirePackage{amsthm}
\RequirePackage{paralist}
\RequirePackage{xspace}
\RequirePackage{xcolor}
\RequirePackage{todonotes}
% \RequirePackage[utf8]{inputenc}
\RequirePackage[english]{babel} % english characters
\RequirePackage{ifmtarg} % convenient to test empty strings
\RequirePackage{datetime} % to change date format
\RequirePackage[left=1in,right=1in,top=1in,bottom=1in]{geometry} % margins 

%% DEFINITION OF OPTIONS <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


\DeclareOption{garamond}{
  \garamondtrue
}
\DeclareOption{times}{
  \timestrue
}

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}

\ProcessOptions\relax

\ifgaramond
\RequirePackage{fontspec}
\setromanfont[Mapping=tex-text,Ligatures={Common},Numbers=Uppercase,Scale=1.05,Style=Historic]{Adobe Garamond Pro}
\fi
\iftimes
\RequirePackage{times}
\RequirePackage{mathptmx}
\fi

\definecolor{stealblue}{rgb}{0.26,0.5,0.70}
\hypersetup{
 %bookmarks=true, % show bookmarks bar?
 % unicode=false, % non-Latin characters in Acrobats bookmarks
 pdftoolbar=true, % show Acrobats toolbar?
 pdfmenubar=true, % show Acrobats menu?
 pdffitwindow=true, % page fit to window when opened
 % pdftitle={ECE 6605 - Information Theory}, % title
 % pdfauthor={Matthieu Bloch}, % author
 pdfsubject={\copyright~Matthieu Bloch \the\year}, % subject of the document
 pdfnewwindow=true, % links in new window
 pdfkeywords={}, % list of keywords
 colorlinks=true, % false: boxed links; true: colored links
 linkcolor={stealblue}, 
 citecolor={stealblue}, 
 urlcolor={stealblue}, 
 %pagecolor={black}, 
 anchorcolor={red} 
}


% line spacing: use to squeeze into 15 pages
\renewcommand{\baselinestretch}{0.96}
\renewcommand{\labelitemi}{\raisebox{0.91pt}{\scalebox{0.5}{$\blacksquare$}}}
% line spacing in itemize and enumerate
\renewenvironment{itemize}{\begin{list}{\labelitemi}{\leftmargin=1.5em\itemsep=0pt\topsep=2pt\partopsep=0pt\parskip=0pt\parsep=2pt}}{\end{list}}
\renewenvironment{enumerate}{\begin{list}{\arabic{enumi}.~}{\usecounter{enumi}\itemsep=2pt\topsep=2pt\partopsep=0pt\parskip=0pt\parsep=2pt}}{\end{list}}

\newcommand{\redTodo}[1]{{\color{red}[#1]}}
\newcommand{\pr}[1]{\left({#1}\right)}

\newcolumntype{P}[1]{>{\centering\arraybackslash}p{#1}}

\makeatletter
\def\@maketitle{%
  \newpage
  \null
  \vspace{-30pt}% avoid wasting space on first page
  \begin{center}%
  \let \footnote \thanks
    {\Large \bf \@title \par}%
    \vskip .25em%
    {\lineskip 0.5em%
      \begin{tabular}[t]{c}%
        \@author
      \end{tabular}\par}%
  \end{center}%
  \par}
\renewcommand\section{\@startsection{section}{1}{\z@}%
                                     {-3ex}%
                                     {1ex}%
                                     {\normalfont\large\bfseries}}
\renewcommand\subsection{\@startsection{subsection}{2}{\z@}%
                                     {-2ex}%
                                     {1ex}%
                                     {\normalfont\normalsize\bfseries}}
\renewcommand\paragraph{\@startsection{paragraph}{4}{\z@}%
                                    {1.5ex}%
                                    {-1em}%
                                    {\normalfont\normalsize\bfseries}}
\long\def\@makecaption#1#2{%
  \vskip\abovecaptionskip
  \sbox\@tempboxa{{\small #1: #2}}%
  \ifdim \wd\@tempboxa >\hsize
  \begin{center}\parbox{0.9\hsize}{\parindent0.0cm%
      {\small #1: #2}\par}\end{center}%
  \else
    \global \@minipagefalse
    \hb@xt@\hsize{\hfil\box\@tempboxa\hfil}%
  \fi
  \vskip\belowcaptionskip}
\makeatother

\def\thepage{\arabic{page}}

\input{arcom_notation}
\input{arcom_acronyms}
\input{arcom_functions}

\newcounter{activitytype}
\setcounter{activitytype}{0}
\newcounter{activitycounter}[activitytype]
\setcounter{activitycounter}{0}
\renewcommand{\theactivitycounter}{\Alph{activitytype}.\arabic{activitycounter}}

\newenvironment{activity}[2][]{\refstepcounter{activitycounter}\begin{center}\begin{mdframed}[linecolor=white,innerleftmargin=0,shadow=false,skipabove=0,skipbelow=0,innertopmargin=0,innerbottommargin=0]\par\noindent{\color{black}\raisebox{0.75pt}{{\boxed{\mbox{\tiny\textsf{{#2} ACTIVITY~\theactivitycounter:~\textsf{#1}}}}}}}~}{\end{mdframed}\end{center}}
\newcommand{\smalltitle}[1]{\raisebox{0.75pt}{{\boxed{\mbox{\tiny\textsf{#1}}}}}}

\definecolor{mygray}{rgb}{.95,0.95,.95}