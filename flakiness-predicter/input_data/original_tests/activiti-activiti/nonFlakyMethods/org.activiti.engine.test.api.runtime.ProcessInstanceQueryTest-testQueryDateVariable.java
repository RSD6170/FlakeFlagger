@Deployment(resources={"org/activiti/engine/test/api/oneTaskProcess.bpmn20.xml"}) public void testQueryDateVariable() throws Exception {
  Map<String,Object> vars=new HashMap<String,Object>();
  Date date1=Calendar.getInstance().getTime();
  vars.put("dateVar",date1);
  ProcessInstance processInstance1=runtimeService.startProcessInstanceByKey("oneTaskProcess",vars);
  Calendar cal2=Calendar.getInstance();
  cal2.add(Calendar.SECOND,1);
  Date date2=cal2.getTime();
  vars=new HashMap<String,Object>();
  vars.put("dateVar",date1);
  vars.put("dateVar2",date2);
  ProcessInstance processInstance2=runtimeService.startProcessInstanceByKey("oneTaskProcess",vars);
  Calendar nextYear=Calendar.getInstance();
  nextYear.add(Calendar.YEAR,1);
  vars=new HashMap<String,Object>();
  vars.put("dateVar",nextYear.getTime());
  ProcessInstance processInstance3=runtimeService.startProcessInstanceByKey("oneTaskProcess",vars);
  Calendar nextMonth=Calendar.getInstance();
  nextMonth.add(Calendar.MONTH,1);
  Calendar twoYearsLater=Calendar.getInstance();
  twoYearsLater.add(Calendar.YEAR,2);
  Calendar oneYearAgo=Calendar.getInstance();
  oneYearAgo.add(Calendar.YEAR,-1);
  ProcessInstanceQuery query=runtimeService.createProcessInstanceQuery().variableValueEquals("dateVar",date1);
  List<ProcessInstance> processInstances=query.list();
  assertNotNull(processInstances);
  assertEquals(2,processInstances.size());
  query=runtimeService.createProcessInstanceQuery().variableValueEquals("dateVar",date1).variableValueEquals("dateVar2",date2);
  ProcessInstance resultInstance=query.singleResult();
  assertNotNull(resultInstance);
  assertEquals(processInstance2.getId(),resultInstance.getId());
  Date unexistingDate=new SimpleDateFormat("dd/MM/yyyy hh:mm:ss").parse("01/01/1989 12:00:00");
  resultInstance=runtimeService.createProcessInstanceQuery().variableValueEquals("dateVar",unexistingDate).singleResult();
  assertNull(resultInstance);
  resultInstance=runtimeService.createProcessInstanceQuery().variableValueNotEquals("dateVar",date1).singleResult();
  assertNotNull(resultInstance);
  assertEquals(processInstance3.getId(),resultInstance.getId());
  resultInstance=runtimeService.createProcessInstanceQuery().variableValueGreaterThan("dateVar",nextMonth.getTime()).singleResult();
  assertNotNull(resultInstance);
  assertEquals(processInstance3.getId(),resultInstance.getId());
  assertEquals(0,runtimeService.createProcessInstanceQuery().variableValueGreaterThan("dateVar",nextYear.getTime()).count());
  assertEquals(3,runtimeService.createProcessInstanceQuery().variableValueGreaterThan("dateVar",oneYearAgo.getTime()).count());
  resultInstance=runtimeService.createProcessInstanceQuery().variableValueGreaterThanOrEqual("dateVar",nextMonth.getTime()).singleResult();
  assertNotNull(resultInstance);
  assertEquals(processInstance3.getId(),resultInstance.getId());
  resultInstance=runtimeService.createProcessInstanceQuery().variableValueGreaterThanOrEqual("dateVar",nextYear.getTime()).singleResult();
  assertNotNull(resultInstance);
  assertEquals(processInstance3.getId(),resultInstance.getId());
  assertEquals(3,runtimeService.createProcessInstanceQuery().variableValueGreaterThanOrEqual("dateVar",oneYearAgo.getTime()).count());
  processInstances=runtimeService.createProcessInstanceQuery().variableValueLessThan("dateVar",nextYear.getTime()).list();
  assertEquals(2,processInstances.size());
  List<String> expectedIds=Arrays.asList(processInstance1.getId(),processInstance2.getId());
  List<String> ids=new ArrayList<String>(Arrays.asList(processInstances.get(0).getId(),processInstances.get(1).getId()));
  ids.removeAll(expectedIds);
  assertTrue(ids.isEmpty());
  assertEquals(0,runtimeService.createProcessInstanceQuery().variableValueLessThan("dateVar",date1).count());
  assertEquals(3,runtimeService.createProcessInstanceQuery().variableValueLessThan("dateVar",twoYearsLater.getTime()).count());
  processInstances=runtimeService.createProcessInstanceQuery().variableValueLessThanOrEqual("dateVar",nextYear.getTime()).list();
  assertEquals(3,processInstances.size());
  assertEquals(0,runtimeService.createProcessInstanceQuery().variableValueLessThanOrEqual("dateVar",oneYearAgo.getTime()).count());
  resultInstance=runtimeService.createProcessInstanceQuery().variableValueEquals(nextYear.getTime()).singleResult();
  assertNotNull(resultInstance);
  assertEquals(processInstance3.getId(),resultInstance.getId());
  processInstances=runtimeService.createProcessInstanceQuery().variableValueEquals(date1).list();
  assertEquals(2,processInstances.size());
  expectedIds=Arrays.asList(processInstance1.getId(),processInstance2.getId());
  ids=new ArrayList<String>(Arrays.asList(processInstances.get(0).getId(),processInstances.get(1).getId()));
  ids.removeAll(expectedIds);
  assertTrue(ids.isEmpty());
  resultInstance=runtimeService.createProcessInstanceQuery().variableValueEquals(twoYearsLater.getTime()).singleResult();
  assertNull(resultInstance);
  runtimeService.deleteProcessInstance(processInstance1.getId(),"test");
  runtimeService.deleteProcessInstance(processInstance2.getId(),"test");
  runtimeService.deleteProcessInstance(processInstance3.getId(),"test");
}